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

parser.add_argument('--ckpt', default='log_train/latest.pth', type=str)

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
# parser.add_argument('-b', '--bitstream', default='output/bitstream/compressed_det_000015.bin', type=str)
# parser.add_argument('-p', '--original_point_cloud', default='output/det_000015.bin', type=str)
parser.add_argument('-b', '--bitstream', default='output/bitstream/compressed_kitti_raw_0000000000.bin', type=str)
parser.add_argument('-p', '--original_point_cloud', default='output/kitti_raw_0000000000.bin', type=str)
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



def decompress():
    lidar_param = LidarParam(args.lidar_yaml)
    H = lidar_param.range_image_height
    W = lidar_param.range_image_width
    
    with open(args.bitstream, 'rb') as f:
        save_content = pickle.load(f)
    
    bitstream_nonground_stage_0 = save_content['nonground_stage_0']
    bitstream_ground_stage_1 = save_content['ground_stage_1']
    bitstream_seg_idx_map = save_content['seg_idx_map']
    bitstream_ac = save_content['bitstream_ac']
    range_image_nonground_stage_0 = basic_compressor.decompress(bitstream_nonground_stage_0)
    range_image_nonground_stage_0 = np.ndarray(shape=(H, W, 1), dtype=np.uint16, buffer=range_image_nonground_stage_0)
    range_image_nonground_stage_0 = range_image_nonground_stage_0 * stage_0_acc
    range_image_ground_stage_1 = basic_compressor.decompress(bitstream_ground_stage_1)
    range_image_ground_stage_1 = np.ndarray(shape=(H, W, 1), dtype=np.uint16, buffer=range_image_ground_stage_1)
    range_image_ground_stage_1 = range_image_ground_stage_1 * stage_1_acc
    seg_idx_map = basic_compressor.decompress(bitstream_seg_idx_map)
    seg_idx_map = np.ndarray(shape=(H, W, 1), dtype=np.int8, buffer=seg_idx_map)
    seg_idx_map = np.copy(seg_idx_map)
    
    range_image_nonground_stage_0 = torch.from_numpy(range_image_nonground_stage_0).float().cuda().unsqueeze(0)
    range_image_ground_stage_1 = torch.from_numpy(range_image_ground_stage_1).float().cuda().unsqueeze(0)
    seg_idx_map = torch.from_numpy(seg_idx_map).int().cuda().unsqueeze(0)
    zero_mask = seg_idx_map != 0
    ground_mask = seg_idx_map == 1
    nonground_mask = seg_idx_map == 2
    
    
    model = RICNet(lidar_param, O_len=stage_1_Q_len, stage_1_bb=args.stage_1_method, stage_2_bb=args.stage_2_method,
                   stage_1_voxel_size=args.stage_1_voxel_size, stage_2_voxel_size=args.stage_2_voxel_size, use_attention=args.use_attention)


    weights, cur_iter, cur_epoch = utils.load_checkpoint(args.ckpt)
    model.load_state_dict(weights)
    
    if torch.cuda.is_available():
        print('Using GPU.')
        model.cuda()
    # TODO: TO add initialize func: Xavier init

    transform_map = model.transform_map
    model.eval()
    
    input_pc_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_nonground_stage_0, transform_map)
    input_ri_stage_1 = range_image_nonground_stage_0
    input_feats_stage_1 = input_ri_stage_1.permute(0, 3, 1, 2)
    prob_stage_1, range_image_pred_stage_1 = model.RICNet_stage_1(input_pc_stage_1,
                                                                input_ri_stage_1,
                                                                input_feats_stage_1,
                                                                stage_1_acc)
    prob_mask_stage_1 = nonground_mask.repeat(1, 1, 1, prob_stage_1.shape[-1])
    
    # ################## without ground 
    # prob_mask_stage_1 = zero_mask.repeat(1, 1, 1, prob_stage_1.shape[-1])
    # ################## without ground
    
    prob_stage_1 = torch.masked_select(prob_stage_1, prob_mask_stage_1).view(-1, stage_1_Q_len)
    # range_image_pred_stage_1 = range_image_pred_stage_1 * nonground_mask * zero_mask + \
                                # range_image_ground_stage_1
    # point_cloud_pred_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_pred_stage_1, transform_map)
    
    # decode residual 
    output_cdf = torchac_utils.pmf_to_cdf(prob_stage_1)
    output_cdf = output_cdf.detach().cpu()
    classification_gt_stage_1 = torchac.decode_float_cdf(output_cdf, bitstream_ac)
    residual_quantized = classification_gt_stage_1 - stage_1_Q_len // 2
    residual = residual_quantized * stage_1_acc
    range_image_nonground_stage_0[nonground_mask] += residual.cuda()
    
    range_image_stage_1 = range_image_nonground_stage_0 + range_image_ground_stage_1
    pc1 = PCTransformer.range_image_to_point_cloud(range_image_nonground_stage_0, transform_map).cpu().numpy()[0]
    pc2 = PCTransformer.range_image_to_point_cloud(range_image_ground_stage_1, transform_map).cpu().numpy()[0]
    compare_point_clouds(pc1, pc2, vis_all=True, \
        save_path=os.path.join('output/temp_vis', 'stage_1.pcd'), save=True, output=False)
    
    point_cloud_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_stage_1, transform_map).cpu().numpy()[0]
    input_pc_stage_2 = torch.from_numpy(point_cloud_stage_1).detach().float().cuda().unsqueeze(0)
    input_ri_stage_2 = range_image_stage_1
    feats = input_ri_stage_2.permute(0, 3, 1, 2)
    

    # TODO: test only stage1
    range_image_pred_stage_2 = model.RICNet_stage_2(input_pc_stage_2,
                                                    input_ri_stage_2,
                                                    feats, stage_1_acc)
    
    # feats = model.stage_2_feature_extractor(input_pc_stage_2, input_ri_stage_2, feats)
    # residual = model.refinement_head(feats)
    # residual = (torch.sigmoid(residual.permute(0, 2, 3, 1)) - 0.5) * stage_1_acc
    # range_image_rec = input_ri_stage_2 - residual
    # point_cloud_pred_stage_2 = PCTransformer.range_image_to_point_cloud(range_image_rec, transform_map).detach().cpu().numpy()[0]

    
    
    
    reconstructed_range_image = range_image_pred_stage_2 * zero_mask
    reconstructed_point_cloud = PCTransformer.range_image_to_point_cloud(reconstructed_range_image, transform_map).detach().cpu().numpy()[0]
    # reconstructed_point_cloud = input_pc_stage_2.detach().cpu().numpy()[0]
    compare_point_clouds(point_cloud_stage_1, reconstructed_point_cloud, vis_all=True, \
        save_path=os.path.join('output/temp_vis', 'stage_2.pcd'), save=True, output=False)

    if args.original_point_cloud is not None:
        
        dataset = Dataset()
        point_cloud_original = dataset.load_pc(args.original_point_cloud)
        range_image = PCTransformer.pointcloud_to_rangeimage(point_cloud_original, lidar_param=lidar_param)
        range_image = np.expand_dims(range_image, -1)
        point_cloud = PCTransformer.range_image_to_point_cloud(range_image, lidar_param.range_image_to_point_cloud_map)
        point_num = np.where(range_image != 0)[0].shape[0]
        
        stage_2_residual = range_image - reconstructed_range_image.detach().cpu().numpy()[0]
        residual_max = np.abs(stage_2_residual).max()
        residual_mean = np.abs(stage_2_residual).sum() / point_num
        
        chamfer_dist = evaluate_metrics.calc_chamfer_distance(point_cloud, reconstructed_point_cloud, out=False)
        point_to_point_result, point_to_plane_result = evaluate_metrics.calc_point_to_point_plane_psnr(point_cloud, reconstructed_point_cloud, out=False)
        
        # bpp and compression rate
        bitstream = basic_compressor.compress(np.round(range_image_stage_1.detach().cpu().numpy()[0] / stage_1_acc).astype(np.uint16).tobytes())
        compressed_size = len(bitstream) * 8
        
        
        # compressed_size = os.path.getsize(args.bitstream) * 8
        
        print('\nCompared with ', args.original_point_cloud)
        print('    BPP: ', compressed_size / point_num)
        print('    Compression Ratio: ', (point_num * 32 * 3) / compressed_size)
        print('    Residual (max): ', residual_max)
        print('    Residual (mean): ', residual_mean)
        print('    Chamfer Distance (mean): ', chamfer_dist['mean'])
        print('    F1 score (threshold=0.02): ', chamfer_dist['f_score'])
        print('    Point-to-Point PSNR (r=59.7): ', point_to_point_result['psnr_mean'])
        print('    Point-to-Plane PSNR (r=59.7): ', point_to_plane_result['psnr_mean'])
    
    IPython.embed()
    
    
    # range_image_nonground_stage_0 = range_image_nonground_stage_0.cpu().numpy()[0]
    # range_image_ground_stage_1 = range_image_ground_stage_1.cpu().numpy()[0]
    # seg_idx_map = seg_idx_map.cpu().numpy()[0]
    # bitstream_nonground_stage_0 = basic_compressor.compress(range_image_nonground_stage_0)
    # bitstream_ground_stage_1 = basic_compressor.compress(range_image_ground_stage_1)
    # bitstream_seg_idx_map = basic_compressor.compress(seg_idx_map)
    # total_len = len(bitstream_nonground_stage_0) + \
    #             len(bitstream_ground_stage_1) + \
    #             len(bitstream_seg_idx_map) + \
    #             len(bitstream_ac)
    
    # ori_file_size = os.path.getsize(args.original_point_cloud)
    # print('compression ratio: ', ori_file_size / total_len)
    
    # # save bitstream to file
    # Path(os.path.join(args.output, 'bitstream')).mkdir(parents=True, exist_ok=True)
    # save_file = os.path.join(args.output, 'bitstream', 'compressed_' + args.original_point_cloud.split('/')[-1])
    
    # save_content = {}
    # save_content['nonground_stage_0'] = bitstream_nonground_stage_0
    # save_content['ground_stage_1'] = bitstream_ground_stage_1
    # save_content['seg_idx_map'] = bitstream_seg_idx_map
    # save_content['bitstream_ac'] = bitstream_ac
    # with open(save_file, 'wb') as f:
    #     pickle.dump(save_content, f)
    # print('save compressed bitstream into binary file ', save_file)
    
    # results_dict = {
    #     'bpp': 0,
    #     'baseline_bpp': 0,
    #     'compression_rate': 0,
    #     'chamfer_distance': 0,
    #     'f1_score': 0,
    #     'point-to-point_psnr': 0,
    #     'point-to-plane_psnr': 0,
    #     'residual_mean': 0,
    #     'residual_max': 0,
    # }

if __name__ == '__main__':
    decompress()
    