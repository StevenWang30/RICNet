import imp
import os
import numpy as np
import argparse

import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Dataset, LidarParam, PCTransformer
from utils.utils import plane_segmentation, calc_plane_residual_depth, calc_plane_residual_vertical

from models.loss import Loss
from models.RICNet import RICNet

import time
import IPython
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from utils import common_utils
from pathlib import Path
from utils import utils
from utils.visualize_utils import visualize_plane_range_image
from utils.evaluate_utils import evaluate_compress_ratio, calc_shannon_entropy
import datetime
from tqdm import tqdm
from utils.visualize_utils import save_point_cloud_to_pcd, compare_point_clouds
from prettytable import PrettyTable
from utils.evaluate_utils import calc_chamfer_distance
from utils.compress_utils import BasicCompressor
from utils import torchac_utils
from utils import evaluate_metrics

import torchac
import pickle

parser = argparse.ArgumentParser()
# Path related arguments
parser.add_argument('--output', default='output/', type=str)

parser.add_argument('--ckpt', default='log_train/latest.pth', type=str)  #/ckpt

# Model related arguments
parser.add_argument('--ground_threshold', default=0.1, type=float)

parser.add_argument('--stage_0_accuracy', default=0.3, type=float)
parser.add_argument('--stage_1_accuracy', default=0.1, type=float)

parser.add_argument('--stage_1_method', default='MinkUNet14', type=str)
parser.add_argument('--stage_2_method', default='MinkUNet101', type=str)
parser.add_argument('--use_attention', default=True, type=bool)

parser.add_argument('--stage_1_voxel_size', default=0.3, type=float)
parser.add_argument('--stage_2_voxel_size', default=0.1, type=float)

# Data related arguments
# parser.add_argument('-p', '--point_cloud', default='output/det_000015.bin', type=str)
parser.add_argument('-p', '--point_cloud', default='output/kitti_raw_0000000000.bin', type=str)
parser.add_argument('--lidar_yaml', default='cfgs/vlp64_KITTI.yaml', type=str)

# Misc arguments
parser.add_argument('--seed', default=123, type=int, help='manual seed')
args = parser.parse_args()

print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))

stage_0_acc = args.stage_0_accuracy * 2
stage_1_acc = args.stage_1_accuracy * 2
stage_1_Q_len = (math.ceil((stage_0_acc / 2) / stage_1_acc)) * 2 + 1

basic_compressor = BasicCompressor(method_name='bzip2')



def compress():
    lidar_param = LidarParam(args.lidar_yaml)
    model = RICNet(lidar_param, O_len=stage_1_Q_len, stage_1_bb=args.stage_1_method, stage_2_bb=args.stage_2_method,
                   stage_1_voxel_size=args.stage_1_voxel_size, stage_2_voxel_size=args.stage_2_voxel_size, use_attention=args.use_attention)

    dataset = Dataset()

    weights, cur_iter, cur_epoch = utils.load_checkpoint(args.ckpt)
    model.load_state_dict(weights)
    
    if torch.cuda.is_available():
        print('Using GPU.')
        model.cuda()
    # TODO: TO add initialize func: Xavier init

    transform_map = model.transform_map
    model.eval()

    point_cloud_original = dataset.load_pc(args.point_cloud)
    range_image = PCTransformer.pointcloud_to_rangeimage(point_cloud_original, lidar_param=lidar_param)
    range_image = np.expand_dims(range_image, -1)
    point_cloud = PCTransformer.range_image_to_point_cloud(range_image, lidar_param.range_image_to_point_cloud_map)
        
    # ground extraction
    pc_filter = point_cloud[np.where(point_cloud[..., 2] < -1.0)]
    if pc_filter.shape[0] < 5000:
        pc_filter = point_cloud[np.where(point_cloud[..., 2] < 0)]
    else:
        random_idx = np.random.choice(pc_filter.shape[0], 5000, replace=False)
        pc_filter = pc_filter[random_idx]
    indices, ground_model = plane_segmentation(pc_filter)
    zero_mask = range_image != 0
    
    point_height = calc_plane_residual_vertical(point_cloud.reshape(-1, 3), ground_model)
    point_height = point_height.reshape(range_image.shape)
    nonground_mask = (point_height > args.ground_threshold) * zero_mask
    ground_mask = (point_height <= args.ground_threshold) * zero_mask
    
    point_cloud = torch.from_numpy(point_cloud).float().cuda().unsqueeze(0)
    range_image = torch.from_numpy(range_image).float().cuda().unsqueeze(0)
    nonground_mask = torch.from_numpy(nonground_mask).bool().cuda().unsqueeze(0)
    ground_mask = torch.from_numpy(ground_mask).bool().cuda().unsqueeze(0)
    zero_mask = torch.from_numpy(zero_mask).bool().cuda().unsqueeze(0)
    
    ground_range_image = range_image * ground_mask
    range_image_ground_stage_1 = torch.round(ground_range_image / stage_1_acc) * stage_1_acc #* ground_mask
    range_image_stage_0 = torch.round(range_image / stage_0_acc) * stage_0_acc
    range_image_nonground_stage_0 = range_image_stage_0 * nonground_mask
    
    # seg_idx_map storage
    seg_idx_map = 0 * (range_image == 0) + 1 * ground_mask + 2 * nonground_mask
    
    # nonground storage 
    nonground_storage = torch.round((range_image * nonground_mask) / stage_0_acc)

    # ground storage
    ground_storage = torch.round(ground_range_image / stage_1_acc)
     
    #********************************** stage 1 **********************************#
    # stage 1: classification for entropy model
    input_pc_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_nonground_stage_0, transform_map)
    input_ri_stage_1 = range_image_nonground_stage_0
    input_feats_stage_1 = input_ri_stage_1.permute(0, 3, 1, 2)
    prob_stage_1, range_image_pred_stage_1 = model.RICNet_stage_1(input_pc_stage_1,
                                                                input_ri_stage_1,
                                                                input_feats_stage_1,
                                                                stage_1_acc)
    prob_mask_stage_1 = (nonground_mask * zero_mask).repeat(1, 1, 1, prob_stage_1.shape[-1])
    
    # ################## without ground 
    # prob_mask_stage_1 = zero_mask.repeat(1, 1, 1, prob_stage_1.shape[-1])
    # ################## without ground
    
    prob_stage_1 = torch.masked_select(prob_stage_1, prob_mask_stage_1).view(-1, stage_1_Q_len)
    range_image_pred_stage_1 = range_image_pred_stage_1 * nonground_mask * zero_mask + \
                                range_image_ground_stage_1
    point_cloud_pred_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_pred_stage_1, transform_map)
    
    residual_quantized = model.quantize((range_image - range_image_stage_0) / stage_1_acc)
    classification_gt_stage_1 = residual_quantized + stage_1_Q_len // 2
    classification_gt_stage_1 = torch.masked_select(classification_gt_stage_1, prob_mask_stage_1[..., [0]])
    assert stage_1_Q_len == prob_stage_1.shape[-1]
    if classification_gt_stage_1.min() < 0 or classification_gt_stage_1.max() >= stage_1_Q_len:
        # point_cloud = (point_cloud * nonground_mask).detach().cpu().numpy()[0]
        # nonground_pc_base = nonground_pc_base.detach().cpu().numpy()[0]
        # compare_point_clouds(point_cloud, nonground_pc_base, vis_all=True,
        #                      save_path=os.path.join(result_dir, 'aa.pcd'), save=True, output=False)
        print('classification_gt_stage_1 class not correct ')
        IPython.embed()
    
    output_cdf = torchac_utils.pmf_to_cdf(prob_stage_1)
    # torchac expects sym as int16, see README for details.
    sym = classification_gt_stage_1.detach().to(torch.int16)
    # torchac expects CDF and sym on CPU.
    output_cdf = output_cdf.detach().cpu()
    sym = sym.detach().cpu()
    # bitstream storage
    bitstream_ac = torchac.encode_float_cdf(output_cdf, sym, check_input_bounds=True)
    
    nonground_storage = nonground_storage.cpu().numpy()[0]
    ground_storage = ground_storage.cpu().numpy()[0]
    seg_idx_map = seg_idx_map.cpu().numpy()[0]
    bitstream_nonground_stage_0 = basic_compressor.compress(nonground_storage.astype(np.uint16).tobytes())
    bitstream_ground_stage_1 = basic_compressor.compress(ground_storage.astype(np.uint16).tobytes())

    bitstream_seg_idx_map = basic_compressor.compress(seg_idx_map.astype(np.int8).tobytes())
    total_len = len(bitstream_nonground_stage_0) + \
                len(bitstream_ground_stage_1) + \
                len(bitstream_seg_idx_map) + \
                len(bitstream_ac)
    # ori_file_size = os.path.getsize(args.point_cloud)
    # print('compression ratio: ', ori_file_size / total_len)
    
    
    range_image_stage_1 = np.round(range_image.cpu().numpy()[0] / stage_1_acc)
    bitstream_stage_1 = basic_compressor.compress(range_image_stage_1.astype(np.uint16).tobytes())
    point_num = np.where(range_image_stage_1 != 0)[0].shape[0]
    
    total_len = len(bitstream_stage_1) + len(bitstream_ac)
    print('bpp: ', total_len * 8 / point_num, ' compression ratio: ', point_num * 3 * 32 / (total_len * 8))
    
    # save bitstream to file
    Path(os.path.join(args.output, 'bitstream')).mkdir(parents=True, exist_ok=True)
    save_file = os.path.join(args.output, 'bitstream', 'compressed_' + args.point_cloud.split('/')[-1])
    
    save_content = {}
    save_content['nonground_stage_0'] = bitstream_nonground_stage_0
    save_content['ground_stage_1'] = bitstream_ground_stage_1
    save_content['seg_idx_map'] = bitstream_seg_idx_map
    save_content['bitstream_ac'] = bitstream_ac
    with open(save_file, 'wb') as f:
        pickle.dump(save_content, f)
    print('save compressed bitstream into binary file ', save_file)
    
    os.path.getsize(save_file)
    print(len(bitstream_nonground_stage_0))
    print(len(bitstream_ground_stage_1))
    print(len(bitstream_seg_idx_map))
    print(len(bitstream_ac))
    # IPython.embed()

if __name__ == '__main__':
    compress()
    