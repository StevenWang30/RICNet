import os
import numpy as np
import argparse

import torch
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dataset import Dataset, LidarParam, PCTransformer
# from network import PCSeg
from models.seg_gan_net import SegGAN
from models.loss import Loss

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

parser = argparse.ArgumentParser()
# Path related arguments
parser.add_argument('--result_dir', default='log_train/test')

parser.add_argument('--reload_flag', default=True)
parser.add_argument('--reload_ckpt_path', default='log_train/ckpt/latest.pth')
# parser.add_argument('--reload_ckpt_path', default='log_train/checkpoint_cluster-100-feats-10-wo-GAN_epoch_17.pth')

# parser.add_argument('--method', default='plane-90-model-2d')
# parser.add_argument('--method', default='test')
parser.add_argument('--cluster_num', default=150)
parser.add_argument('--feats_num', default=0)
# parser.add_argument('--plane_num', default=2)

# Model related arguments
parser.add_argument('--max_epoch', default=40)
parser.add_argument('--batch_size', default=2)
parser.add_argument('--log_dir', default='log_train/train/tensorboard')
parser.add_argument('--save_weight_dir', default='log_train/ckpt')
parser.add_argument('--learning_rate', default=0.01)
parser.add_argument('--USE_GAN', default=False)

parser.add_argument('--stage_0_accuracy', default=0.5)
parser.add_argument('--stage_1_accuracy', default=0.1)
parser.add_argument('--stage_2_accuracy', default=0.02)

parser.add_argument('--stage_1_method', default='MinkUNet14')
parser.add_argument('--stage_2_method', default='MinkUNet14')

parser.add_argument('--stage_1_voxel_size', default=0.2)
parser.add_argument('--stage_2_voxel_size', default=0.1)

# Data related arguments
parser.add_argument('--train_datalist', default='data/train_64E_KITTI.txt')
# parser.add_argument('--train_datalist', default='data/val_64E_KITTI.txt')
parser.add_argument('--val_datalist', default='data/val_64E_KITTI.txt')
parser.add_argument('--test_datalist', default=None, help='check in the test() function code.')
parser.add_argument('--lidar_yaml', default='cfgs/vlp64_KITTI.yaml')
# parser.add_argument('--use_radius_outlier_removal', default=True)

parser.add_argument('--test', action='store_true', help='default False. Add --test sets True.')

# Misc arguments
parser.add_argument('--seed', default=123, type=int, help='manual seed')
parser.add_argument('--grad_norm_clip', default=10)

parser.add_argument('--clean_tb_summary', default=True)

args = parser.parse_args()

# args.method = 'cluster-' + str(args.cluster_num) + '-feats-' + str(args.feats_num)
# if args.USE_GAN:
#     args.method += '-GAN' # + '-model-' + args.model_type
# else:
#     args.method += '-wo-GAN'
args.method = '2-small-kernel-prob-detach-ground-3-stage' + \
              '-' + args.stage_1_method + '-' + args.stage_2_method + \
              '-cluster-' + str(int(args.cluster_num)) + '-feats-' + str(int(args.feats_num)) + \
              '-voxelsize-' + str(int(args.stage_1_voxel_size * 100)) + '-' + str(int(args.stage_2_voxel_size * 100)) + \
              '-acc-' + str(int(args.stage_0_accuracy * 100)) + '-' + str(int(args.stage_1_accuracy * 100)) + '-' + str(int(args.stage_2_accuracy * 100))

args.method = 'cvpr-all-with-attention'

print("Input arguments:")
for key, val in vars(args).items():
    print("{:16} {}".format(key, val))


result_dir = os.path.join(args.result_dir, args.method)
Path(result_dir).mkdir(parents=True, exist_ok=True)

train_cr = 0
val_cr = 0

COLOR_MAP = np.random.random((args.cluster_num + 1, 3))  # np.random.random((args.cluster_num + args.plane_num, 3))
# COLOR_MAP[args.cluster_num:] = [1, 0, 0]
# COLOR_MAP[:, :] = [0, 0, 1]
COLOR_MAP[:2] = [1, 0, 0]

# rec_acc = 100 / args.cluster_num * 30
stage_0_acc = args.stage_0_accuracy * 2
stage_1_acc = args.stage_1_accuracy * 2
stage_2_acc = args.stage_2_accuracy * 2
stage_1_Q_len = (math.ceil((stage_0_acc / 2) / stage_1_acc) + 1) * 2
stage_2_Q_len = (math.ceil((stage_1_acc / 2) / stage_2_acc) + 1) * 2

basic_compressor = BasicCompressor(method_name='bzip2')


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def train():
    max_epoch = args.max_epoch
    train_tb_summary = SummaryWriter(train_tb_dir)
    val_tb_summary = SummaryWriter(val_tb_dir)

    lr = args.learning_rate  # if epoch < 75 else 0.0001

    lidar_param = LidarParam(args.lidar_yaml)
    train_dataset = Dataset(datalist=args.train_datalist, lidar_param=lidar_param, plane_seg=True, aug=True)
    val_dataset = Dataset(datalist=args.val_datalist, lidar_param=lidar_param, plane_seg=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print('load data finished.')
    # val_loader = DataLoader(val_data_generate, batch_size=args.batch_size, shuffle=True)

    model = SegGAN(lidar_param, cluster_num=args.cluster_num, feats_num=args.feats_num,
                   stage_1_Q_len=stage_1_Q_len, stage_2_Q_len=stage_2_Q_len,
                   stage_1_method=args.stage_1_method, stage_2_method=args.stage_2_method,
                   stage_1_voxel_size=args.stage_1_voxel_size, stage_2_voxel_size=args.stage_2_voxel_size)
    loss_fn = Loss()

    # count_parameters(model)

    cur_iter = 0
    cur_epoch = 0
    if args.reload_flag:
        # reload_path = os.path.join(args.save_weight_dir, str(args.reload_epoch) + '.pth')
        # model.load_state_dict(torch.load(args.reload_ckpt_path))
        weights, cur_iter, cur_epoch = utils.load_checkpoint(args.reload_ckpt_path)
        model.load_state_dict(weights)
    init_epoch = cur_epoch + 1

    if torch.cuda.is_available():
        print('Using GPU.')
        model.cuda()
    # TODO: TO add initialize func: Xavier init

    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Optimizers
    param_G = list(model.generator.parameters()) + list(model.cluster_extractor.parameters())
    param_D = list(model.discriminator.parameters())
    optimizer_G = torch.optim.Adam(param_G, lr=lr)
    optimizer_D = torch.optim.Adam(param_D, lr=lr)
    optimizer = (optimizer_G, optimizer_D)

    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=steps_per_epoch, epochs=max_epoch)
    scheduler = None

    for epoch in tqdm(range(init_epoch, init_epoch + max_epoch)):
        train_dataset = Dataset(datalist=args.train_datalist, lidar_param=lidar_param, plane_seg=True,
                                random_sample=1600)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

        # train
        cur_iter = iter_one_epoch(model, loss_fn, train_loader, cur_iter, train_tb_summary, optimizer, scheduler,
                                  train=True)

        cur_epoch = cur_iter // len(train_dataset)
        if cur_epoch % 5 == 0 and cur_epoch > 0:
            utils.save_checkpoint(model, args.save_weight_dir, args.method, cur_iter, cur_epoch, max_ckpt_save_num=1)
            # torch.save(model.state_dict(), os.path.join(args.save_weight_dir, "latest.pth"))
        utils.save_checkpoint(model, args.save_weight_dir, 'latest', cur_iter, cur_epoch, single=True)

        # validation
        with torch.no_grad():
            iter_one_epoch(model, loss_fn, val_loader, cur_iter, val_tb_summary, log_path=log_txt_path, train=False)

    # model_save = os.path.join('log_train/ckpt_save', args.method + '-epoch' + str(max_epoch) + '.pth')
    # torch.save(model.state_dict(), model_save)
    print('save final model weight')
    utils.save_checkpoint(model, 'log_train/ckpt_save', args.method + '-epoch' + str(max_epoch), cur_iter, max_epoch,
                          single=True)

    train_tb_summary.close()
    val_tb_summary.close()


def test():
    test_tb_dir = os.path.join(args.log_dir, args.method, 'test')
    Path(test_tb_dir).mkdir(parents=True, exist_ok=True)
    test_tb_summary = SummaryWriter(test_tb_dir)
    lidar_param = LidarParam(args.lidar_yaml)

    model = SegGAN(lidar_param, cluster_num=args.cluster_num, feats_num=args.feats_num, stage_1_Q_len=stage_1_Q_len,
                   stage_2_Q_len=stage_2_Q_len)
    if torch.cuda.is_available():
        model.cuda()
    loss_fn = Loss()

    weights, cur_iter = utils.load_checkpoint(args.reload_ckpt_path)
    model.load_state_dict(weights)

    test_datalist = [
        'data/test_64E_KITTI_city.txt',
        'data/test_64E_KITTI_campus.txt',
        'data/test_64E_KITTI_road.txt',
        'data/test_64E_KITTI_residential.txt',
    ]
    for idx, datalist in enumerate(test_datalist):
        cur_iter = idx + 1
        test_dataset = Dataset(datalist=datalist, lidar_param=lidar_param, plane_seg=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            iter_one_epoch(model, loss_fn, test_loader, cur_iter, test_tb_summary, log_path=log_txt_path, train=False,
                           full_eval=False)
    test_tb_summary.close()


def iter_one_epoch(model, loss_fn, data_loader, cur_iter, tb_summary, optimizer=None, scheduler=None, log_path=None,
                   train=True, full_eval=True):
    transform_map = model.transform_map
    if train:
        model.train()
        optimizer_G, optimizer_D = optimizer
        eval_interval = 50
    else:
        model.eval()
        eval_interval = 1

    pbar = tqdm(total=len(data_loader))
    for i, (point_cloud, range_image, ground_model) in enumerate(data_loader):
        batch_size = point_cloud.shape[0]
        if train and range_image.shape[0] < batch_size:
            # print('range_image.shape[0] < batch_size')
            # IPython.embed()
            continue
        if train:
            cur_iter = cur_iter + batch_size
        torch.cuda.synchronize()
        t = time.time()

        
        range_image = range_image.float().cuda()
        point_cloud = point_cloud.float().cuda()
        # ground_mask = ground_mask.float().cuda()
        # nonground_mask = nonground_mask.float().cuda()
        ground_model = ground_model.float().cuda()
        zero_mask = range_image != 0

        # -----------------
        #  Train Generator
        # -----------------
        if train:
            optimizer_G.zero_grad()

        cluster_xyz, cluster_feats, seg_idx_map = model.compress_point_cloud(point_cloud, range_image, ground_model,
                                                                             stage_0_acc)

        ground_mask = (seg_idx_map == 0).unsqueeze(-1)
        nonground_mask = (seg_idx_map != 0).unsqueeze(-1)
        nonground_range_image = range_image * nonground_mask
        ground_range_image = range_image * ground_mask

        # torch.cuda.synchronize()
        # print('compress time cost:', time.time() - t)
        # t = time.time()

        # point_cloud_base, range_image_base, feats_base = model.reconstruct_point_cloud(cluster_xyz, cluster_feats, seg_idx_map)
        nonground_pc_base, nonground_ri_base, nonground_feats_base = \
            model.reconstruct_nonground_point_cloud(cluster_xyz, cluster_feats, seg_idx_map - 1, nonground_mask)
        ground_pc_base, ground_ri_base = model.reconstruct_ground_point_cloud(ground_model, ground_mask)

        ground_residual = ground_range_image - ground_ri_base
        ground_residual_quantized = torch.round(ground_residual / stage_1_acc)

        baseline_ground_residual_quantized = torch.round((ground_range_image - ground_ri_base) / stage_2_acc)
        baseline_nonground_residual_quantized = torch.round((nonground_range_image - nonground_ri_base) / stage_2_acc)
        baseline_residual_quantized = torch.round((range_image - ground_ri_base - nonground_ri_base) / stage_2_acc)

        range_image_pred_stage_0 = nonground_ri_base
        point_cloud_pred_stage_0 = nonground_pc_base
        residual_stage_0 = nonground_range_image - range_image_pred_stage_0
        residual_stage_0_quantized = torch.round(residual_stage_0 / stage_0_acc)
        range_image_rec_stage_0 = range_image_pred_stage_0 + residual_stage_0_quantized * stage_0_acc
        point_cloud_rec_stage_0 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_0, transform_map)

        # torch.cuda.synchronize()
        # print('reconstruct time cost:', time.time() - t)
        # t = time.time()

        # feats = nonground_feats_base
        feats = range_image_rec_stage_0.permute(0, 3, 1, 2)
        prob_stage_1, range_image_pred_stage_1, feats_stage_1 = model.refine_point_cloud(point_cloud_rec_stage_0,
                                                                                         range_image_rec_stage_0,
                                                                                         feats,
                                                                                         stage_1_acc, stage=1)
        prob_mask_stage_1 = (nonground_mask * zero_mask).repeat(1, 1, 1, prob_stage_1.shape[-1])
        prob_stage_1 = torch.masked_select(prob_stage_1, prob_mask_stage_1).view(-1, stage_1_Q_len)
        range_image_pred_stage_1 = range_image_pred_stage_1 * nonground_mask * zero_mask + \
                                   ground_residual_quantized * stage_1_acc

        feats_stage_1 = feats_stage_1 * nonground_mask.permute(0, 3, 1, 2) * zero_mask.permute(0, 3, 1, 2)
        residual_quantized = model.quantize((nonground_range_image - range_image_rec_stage_0) / stage_1_acc)
        classification_gt_stage_1 = residual_quantized + stage_1_Q_len // 2
        classification_gt_stage_1 = torch.masked_select(classification_gt_stage_1, prob_mask_stage_1[..., [0]])
        # # without grad
        # range_image_rec_stage_1 = range_image_rec_stage_0 + residual_quantized * stage_1_acc

        # # with grad
        # residual_quantized_with_grad = model.quantize((range_image - range_image_pred_stage_1) / stage_1_acc)
        # range_image_rec_stage_1 = range_image_pred_stage_1 + residual_quantized_with_grad * stage_1_acc

        # # without grad
        # range_image_rec_stage_1 = range_image_rec_stage_1.detach()
        range_image_rec_stage_1 = model.quantize(range_image / stage_1_acc) * stage_1_acc

        point_cloud_rec_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_1, transform_map)
        point_cloud_pred_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_pred_stage_1, transform_map)

        # ground_residual_stage_1 = ground_range_image - ground_ri_base
        # ground_residual_stage_1_quantized = torch.round(ground_residual_stage_1 / stage_1_acc)
        #
        # prob_stage_1, range_image_pred_stage_1, feats_stage_1 = model.refine_point_cloud(nonground_pc_base,
        #                                                                             nonground_ri_base,
        #                                                                             nonground_feats_base, stage_1_acc, stage=1)
        # range_image_pred_stage_1 = (range_image_pred_stage_1 * nonground_mask + ground_ri_base) * zero_mask
        # feats_stage_1 = feats_stage_1 * nonground_mask.permute(0, 3, 1, 2)
        #
        # nonground_residual_quantized = model.quantize((nonground_range_image - nonground_ri_base) / stage_1_acc)
        # classification_gt_stage_1 = nonground_residual_quantized + stage_1_Q_len // 2
        #
        # # # without gradient
        # # range_image_rec_stage_1 = nonground_ri_base + nonground_residual_quantized * stage_1_acc + \
        # #                           ground_ri_base + ground_residual_stage_1_quantized * stage_1_acc
        #
        # # with gradient for backward
        # residual_quantized_with_grad = model.quantize((range_image - range_image_pred_stage_1) / stage_1_acc)
        # range_image_rec_stage_1 = range_image_pred_stage_1 + residual_quantized_with_grad * stage_1_acc
        #
        # # residual_quantized = model.quantize((range_image - range_image_pred_stage_1) / stage_1_acc)
        # # classification_gt_stage_1 = residual_quantized + stage_1_Q_len // 2
        # # range_image_rec_stage_1 = range_image_pred_stage_1 + residual_quantized * stage_1_acc
        # point_cloud_rec_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_1, transform_map)
        # point_cloud_pred_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_pred_stage_1, transform_map)

        assert stage_1_Q_len == prob_stage_1.shape[-1]
        if classification_gt_stage_1.min() < 0 or classification_gt_stage_1.max() >= stage_1_Q_len:
            # point_cloud = (point_cloud * nonground_mask).detach().cpu().numpy()[0]
            # nonground_pc_base = nonground_pc_base.detach().cpu().numpy()[0]
            # compare_point_clouds(point_cloud, nonground_pc_base, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'aa.pcd'), save=True, output=False)
            print('classification_gt_stage_1 class not correct ')
            IPython.embed()

        # torch.cuda.synchronize()
        # print('stage 1 refine time cost:', time.time() - t)
        # t = time.time()

        # feats = torch.cat((point_cloud_rec_stage_1, range_image_rec_stage_1), -1)
        # feats = feats.permute(0, 3, 1, 2)
        # feats = torch.cat((feats, nonground_feats_base), 1)
        # feats = torch.cat((feats, feats_stage_1), 1)
        
        feats = range_image_rec_stage_1.permute(0, 3, 1, 2)
        # TODO: test only stage1
        prob_stage_2, range_image_pred_stage_2, feats_stage_2 = model.refine_point_cloud(point_cloud_rec_stage_1,
                                                                                         range_image_rec_stage_1,
                                                                                         feats, stage_2_acc, stage=2)
        # residual_quantized = model.quantize((range_image - range_image_pred_stage_2) / stage_2_acc)
        # classification_gt_stage_2 = residual_quantized + stage_2_Q_len // 2
        # range_image_rec_stage_2 = range_image_pred_stage_2 + residual_quantized * stage_2_acc
        prob_mask_stage_2 = zero_mask.repeat(1, 1, 1, prob_stage_2.shape[-1])
        prob_stage_2 = torch.masked_select(prob_stage_2, prob_mask_stage_2).view(-1, stage_2_Q_len)

        range_image_pred_stage_2 *= zero_mask
        residual_quantized = model.quantize((range_image - range_image_rec_stage_1) / stage_2_acc)
        classification_gt_stage_2 = residual_quantized + stage_2_Q_len // 2
        classification_gt_stage_2 = torch.masked_select(classification_gt_stage_2, prob_mask_stage_2[..., [0]])
        range_image_rec_stage_2 = range_image_rec_stage_1 + residual_quantized * stage_2_acc
        point_cloud_rec_stage_2 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_2, transform_map)
        point_cloud_pred_stage_2 = PCTransformer.range_image_to_point_cloud(range_image_pred_stage_2, transform_map)
        assert stage_2_Q_len == prob_stage_2.shape[-1]
        if classification_gt_stage_2.min() < 0 or classification_gt_stage_2.max() >= stage_2_Q_len:
            print('classification_gt_stage_2 class not correct ')
            IPython.embed()
        # IPython.embed()
        # a = model.quantize((range_image - range_image_pred_stage_2) / stage_2_acc)
        # b = range_image_pred_stage_2 + a * stage_2_acc
        # c, d = loss_fn.bitrate_loss(prob_stage_2, a.detach())
        # c, d = loss_fn.bitrate_loss(prob_stage_2, (classification_gt_stage_2 * zero_mask).detach())

        # torch.cuda.synchronize()
        # print('stage 2 refine time cost:', time.time() - t)
        # t = time.time()

        # reconstrct l2 loss
        reconstruct_loss_stage_1 = loss_fn.l2_residual_loss(range_image_pred_stage_1, range_image, nonground_mask * zero_mask)
        tb_summary.add_scalar('loss/reconstruct_loss_stage_1', reconstruct_loss_stage_1.item(), cur_iter)
        # reconstruct_base_loss = loss_fn.l2_residual_loss(range_image_base, range_image)
        # TODO: test only stage1
        reconstruct_loss_stage_2 = loss_fn.l2_residual_loss(range_image_pred_stage_2, range_image, zero_mask)
        tb_summary.add_scalar('loss/reconstruct_loss_stage_2', reconstruct_loss_stage_2.item(), cur_iter)

        # # chamfer distance loss
        # chamfer_dist_loss_stage_1 = loss_fn.chamfer_distance_loss(point_cloud.view(batch_size, -1, 3),
        #                                                           point_cloud_pred_stage_1.view(batch_size, -1, 3),
        #                                                           nonground_mask.view(batch_size, -1, 1))
        # chamfer_dist_loss_stage_2 = loss_fn.chamfer_distance_loss(point_cloud.view(batch_size, -1, 3),
        #                                                           point_cloud_pred_stage_2.view(batch_size, -1, 3),
        #                                                           nonground_mask.view(batch_size, -1, 1))

        # classification cross entropy loss
        classification_loss_stage_1 = loss_fn.classification_loss(prob_stage_1, classification_gt_stage_1.detach())
        tb_summary.add_scalar('loss/classification_loss_stage_1', classification_loss_stage_1.item(), cur_iter)
        estimate_bitrate_stage_1, real_bitrate_stage_1 = loss_fn.bitrate_loss(prob_stage_1,
                                                                              classification_gt_stage_1.detach())
        bpp_loss_stage_1 = estimate_bitrate_stage_1 / (64 * 2048 * 3) / batch_size
        tb_summary.add_scalar('loss/bpp_loss_stage_1', bpp_loss_stage_1.item(), cur_iter)
        tb_summary.add_scalar('compress_len/real_byte_len_stage_1', real_bitrate_stage_1 / batch_size / 8, cur_iter)
        # tb_summary.add_scalar('compress_len/estimate_byte_len_stage_1', estimate_bitrate_stage_1 / batch_size / 8,
        #                       cur_iter)

        # TODO: test only stage1
        classification_loss_stage_2 = loss_fn.classification_loss(prob_stage_2, classification_gt_stage_2.detach())
        tb_summary.add_scalar('loss/classification_loss_stage_2', classification_loss_stage_2.item(), cur_iter)
        estimate_bitrate_stage_2, real_bitrate_stage_2 = loss_fn.bitrate_loss(prob_stage_2,
                                                                              classification_gt_stage_2.detach())
        bpp_loss_stage_2 = estimate_bitrate_stage_2 / (64 * 2048 * 3)
        tb_summary.add_scalar('loss/bpp_loss_stage_2', bpp_loss_stage_2.item(), cur_iter)
        tb_summary.add_scalar('compress_len/real_byte_len_stage_2', real_bitrate_stage_2 / batch_size / 8, cur_iter)
        # tb_summary.add_scalar('compress_len/estimate_byte_len_stage_2', estimate_bitrate_stage_2 / batch_size / 8,
        #                       cur_iter)

        residual_baseline_stage_1 = nonground_range_image - range_image_rec_stage_0
        residual_rec_stage_1 = nonground_range_image - range_image_pred_stage_1
        # residual_base = range_image - range_image_base
        # TODO: test only stage1
        residual_baseline_stage_2 = range_image - range_image_rec_stage_1
        residual_rec_stage_2 = range_image - range_image_pred_stage_2

        # tb_summary.add_scalar('residual/max_reconstruct', residual_rec.abs().max(), cur_iter)
        # tb_summary.add_scalar('residual/max_baseline', residual_clt.abs().max(), cur_iter)
        # tb_summary.add_scalar('residual/mean_nonground_reconstruct_stage_1',
        #                       (residual_rec * nonground_mask).abs().sum() / nonground_mask.sum(), cur_iter)
        # tb_summary.add_scalar('residual/mean_nonground_baseline_stage_1',
        #                       (residual_base * nonground_mask).abs().sum() / nonground_mask.sum(), cur_iter)
        tb_summary.add_scalar('residual/mean_reconstruct_stage_1', residual_rec_stage_1.abs().mean(), cur_iter)
        # tb_summary.add_scalar('residual/mean_baseline_stage_1', residual_base.abs().mean(), cur_iter)
        # TODO: test only stage1
        tb_summary.add_scalar('residual/mean_reconstruct_baseline_stage_2', residual_baseline_stage_2.abs().mean(),
                              cur_iter)
        tb_summary.add_scalar('residual/mean_reconstruct_stage_2', residual_rec_stage_2.abs().mean(), cur_iter)
        tb_summary.add_scalar('residual/mean_cmr_stage_2',
                              residual_rec_stage_2.abs().mean() / residual_baseline_stage_2.abs().mean(), cur_iter)

        loss_G = classification_loss_stage_1 + bpp_loss_stage_1 + reconstruct_loss_stage_1 + \
                 reconstruct_loss_stage_2  #classification_loss_stage_2 + bpp_loss_stage_2 +  + chamfer_dist_loss_stage_2  # +  reconstruct_loss_stage_2  #  +  + reconstruct_lossentropy_loss + 0.1 *
        # 0.01 * chamfer_dist_loss_stage_1 +\
        
        # loss_G = classification_loss_stage_2

        if args.USE_GAN:
            # Loss measures generator's ability to fool the discriminator
            feats = torch.cat((point_cloud_pred_stage_2, range_image_pred_stage_2), -1)
            fake_pred = model.discriminator_validity(point_cloud_pred_stage_2, feats)
            g_loss = loss_fn.adversarial_loss(fake_pred, valid=True)
            tb_summary.add_scalar('loss/g_loss', g_loss.item(), cur_iter)
            loss_G = loss_G + g_loss

        if torch.isnan(loss_G):
            print('loss_G is nan!!!')
            IPython.embed()
        tb_summary.add_scalar('loss/loss_G', loss_G.item(), cur_iter)

        # torch.cuda.synchronize()
        # print('G loss time cost:', time.time() - t)
        # t = time.time()

        if train:
            loss_G.backward()
            clip_grad_norm_(model.parameters(), args.grad_norm_clip)
            optimizer_G.step()

        # torch.cuda.synchronize()
        # print('G backward time cost:', time.time() - t)
        # t = time.time()

        if args.USE_GAN:
            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            feats = torch.cat((point_cloud_pred_stage_2, range_image_pred_stage_2), -1)
            fake_pred = model.discriminator_validity(point_cloud_pred_stage_2.detach(), feats.detach())
            d_loss_fake = loss_fn.adversarial_loss(fake_pred, valid=False)
            feats = torch.cat((point_cloud, range_image), -1)
            real_pred = model.discriminator_validity(point_cloud, feats)
            d_loss_real = loss_fn.adversarial_loss(real_pred, valid=True)
            tb_summary.add_scalar('validity/fake_pred', fake_pred.mean(), cur_iter)
            tb_summary.add_scalar('validity/real_pred', real_pred.mean(), cur_iter)
            tb_summary.add_scalar('loss/d_loss_fake', d_loss_fake.item(), cur_iter)
            tb_summary.add_scalar('loss/d_loss_real', d_loss_real.item(), cur_iter)

            loss_D = (d_loss_fake + d_loss_real) / 2
            tb_summary.add_scalar('loss/loss_D', loss_D.item(), cur_iter)
            if train:
                loss_D.backward()
                clip_grad_norm_(model.parameters(), args.grad_norm_clip)
                optimizer_D.step()

        if train:
            if scheduler is not None:
                scheduler.step()

            cur_lr_G = optimizer_G.param_groups[0]['lr']
            tb_summary.add_scalar('zmeta/learning_rate_G', cur_lr_G, cur_iter)

            cur_lr_D = optimizer_D.param_groups[0]['lr']
            tb_summary.add_scalar('zmeta/learning_rate_D', cur_lr_D, cur_iter)
        # torch.cuda.synchronize()
        # print('in model loss: ', time.clock() - a)
        # a = time.clock()
        # if i % 30 == 0:
        #     g = model.cluster_extractor.out[0].weight.grad
        #     print('cluster_extractor g min: ', g.min(), '  g max: ', g.max())
        #     # IPython.embed()
        #     g = model.generator.stage_1_net.refine_net.conv1.conv.weight.grad # 2d easy
        #     # g = model.generator.stage_1_net.refine_net.SA_modules[0].mlps[0][0].weight.grad
        #     print('generator g min: ', g.min(), '  g max: ', g.max())
        #     # IPython.embed()

        # point_cloud = point_cloud.cpu().numpy()[0]
        # point_cloud_rec_stage_0 = point_cloud_rec_stage_0.detach().cpu().numpy()[0]
        # point_cloud_rec_stage_1 = point_cloud_rec_stage_1.detach().cpu().numpy()[0]
        # point_cloud_rec_stage_2 = point_cloud_rec_stage_2.detach().cpu().numpy()[0]
        # point_cloud_pred_stage_1 = point_cloud_pred_stage_1.detach().cpu().numpy()[0]
        # point_cloud_pred_stage_2 = point_cloud_pred_stage_2.detach().cpu().numpy()[0]
        # compare_point_clouds(point_cloud, point_cloud_pred_stage_1, vis_all=True,
        #                      save_path=os.path.join(result_dir, 'pc_pred_1.pcd'), save=True, output=False)
        # compare_point_clouds(point_cloud, point_cloud_pred_stage_2, vis_all=True,
        #                      save_path=os.path.join(result_dir, 'pc_pred_2.pcd'), save=True, output=False)
        # compare_point_clouds(point_cloud_rec_stage_1, point_cloud_pred_stage_2, vis_all=True,
        #                      save_path=os.path.join(result_dir, 'pc_cmp_2.pcd'), save=True, output=False)
        # compare_point_clouds(point_cloud_rec_stage_0, point_cloud_pred_stage_1, vis_all=True,
        #                      save_path=os.path.join(result_dir, 'pc_cmp_1.pcd'), save=True, output=False)

        # # len(basic_compressor.compress(np.round(residual_rec_stage_2 / stage_1_acc).astype(np.int16))
        # ground_residual = range_image * ground_mask - ground_ri_base
        # ground_residual = ground_residual.detach().cpu().numpy()[0]
        # print(
        #     len(basic_compressor.compress(np.round(ground_residual / stage_2_acc).astype(np.int16)))
        # )
        # IPython.embed()

        if i % eval_interval == 0:
            # compression_result_dict_stage_1_baseline, original_data_stage_1_baseline, compressed_data_stage_1_baseline = \
            #     evaluate_compress_ratio(point_cloud, range_image, seg_idx_map, residual_base, None, cluster_xyz, cluster_feats, stage_1_acc)
            # for key, value in compression_result_dict_stage_1_baseline.items():
            #     tb_summary.add_scalar('z_compress_eval_metrics_details_baseline/' + key, value, cur_iter)

            compression_result_dict_stage_0, original_data_stage_0, compressed_data_stage_0 = \
                evaluate_compress_ratio(point_cloud, range_image, seg_idx_map, None,
                                        None, cluster_xyz, cluster_feats, stage_0_acc)
            # stage_0_rec_bitrate = len(compressed_data_stage_0['residual_unit']) * 8
            info_bitrate = compression_result_dict_stage_0['total_bitrate']  # - stage_0_rec_bitrate
            tb_summary.add_scalar('compress_len/info_compress_len_rec', info_bitrate / 8, cur_iter)

            residual_stage_0_quantized = residual_stage_0_quantized.cpu().numpy()[0]
            stage_0_rec_bitrate = len(basic_compressor.compress(residual_stage_0_quantized.astype(np.int16))) * 8
            tb_summary.add_scalar('compress_len/residual_compress_len_stage_0_rec', stage_0_rec_bitrate / 8, cur_iter)

            ground_residual_quantized = ground_residual_quantized.cpu().numpy()[0]
            ground_residual_bitrate = len(basic_compressor.compress(ground_residual_quantized.astype(np.int16))) * 8
            tb_summary.add_scalar('compress_len/residual_compress_len_ground', ground_residual_bitrate / 8, cur_iter)

            if full_eval:
                for key, value in compression_result_dict_stage_0.items():
                    tb_summary.add_scalar('z_compress_eval_info_details_metrics/' + key, value, cur_iter)

                baseline_ground_residual_quantized = baseline_ground_residual_quantized.cpu().numpy()[0]
                baseline_ground_residual_bitrate = len(
                    basic_compressor.compress(baseline_ground_residual_quantized.astype(np.int16))) * 8
                tb_summary.add_scalar('compress_len/residual_compress_len_baseline_ground',
                                      baseline_ground_residual_bitrate / 8, cur_iter)

                baseline_nonground_residual_quantized = baseline_nonground_residual_quantized.cpu().numpy()[0]
                baseline_nonground_residual_bitrate = len(
                    basic_compressor.compress(baseline_nonground_residual_quantized.astype(np.int16))) * 8
                tb_summary.add_scalar('compress_len/residual_compress_len_baseline_nonground',
                                      baseline_nonground_residual_bitrate / 8, cur_iter)

                baseline_residual_quantized = baseline_residual_quantized.cpu().numpy()[0]
                baseline_residual_bitrate = len(
                    basic_compressor.compress(baseline_residual_quantized.astype(np.int16))) * 8
                tb_summary.add_scalar('compress_len/residual_compress_len_baseline_all', baseline_residual_bitrate / 8,
                                      cur_iter)

                residual_baseline_stage_2 = residual_baseline_stage_2.detach().cpu().numpy()[0]
                tb_summary.add_scalar('compress_len/residual_compress_len_stage_2_baseline',
                                      len(basic_compressor.compress(
                                          np.round(residual_baseline_stage_2 / stage_2_acc).astype(np.int16))),
                                      cur_iter)
                # residual_stage_0_quantized = residual_stage_0_quantized.cpu().numpy()[0]
                residual_baseline_stage_1 = residual_baseline_stage_1.detach().cpu().numpy()[0]
                tb_summary.add_scalar('compress_len/residual_compress_len_stage_1_baseline',
                                      len(basic_compressor.compress(
                                          np.round(residual_baseline_stage_1 / stage_1_acc).astype(np.int16))),
                                      cur_iter)
                residual_rec_stage_1 = residual_rec_stage_1.detach().cpu().numpy()[0]
                tb_summary.add_scalar('compress_len/residual_compress_len_stage_1_pred',
                                      len(basic_compressor.compress(
                                          np.round(residual_rec_stage_1 / stage_1_acc).astype(np.int16))), cur_iter)
                residual_rec_stage_2 = residual_rec_stage_2.detach().cpu().numpy()[0]
                tb_summary.add_scalar('compress_len/residual_compress_len_stage_2_pred',
                                      len(basic_compressor.compress(
                                          np.round(residual_rec_stage_2 / stage_2_acc).astype(np.int16))), cur_iter)

            # tb_summary.add_scalar('compress_len/residual_compress_len_baseline', len(compressed_data_baseline['residual_unit']),
            #                       cur_iter)
            # tb_summary.add_scalar('compress_len/residual_compress_cmp',
            #                       len(compressed_data_baseline['residual_unit']) /
            #                       (len(compressed_stage_2_data['residual_unit']) +
            #                        len(compressed_data['residual_unit'])), cur_iter)
            # tb_summary.add_scalar('compress_len/residual_compress_stage_1_ours_baseline',
            #                       real_bitrate_stage_1 / batch_size /
            #                       len(compressed_data_stage_1_baseline['residual_unit']), cur_iter)
            # # TODO: test only stage1
            # tb_summary.add_scalar('compress_len/residual_compress_stage_2_ours_baseline',
            #                       real_bitrate_stage_2 / batch_size /
            #                       len(compressed_data_stage_2_baseline['residual_unit']), cur_iter)

            seg_idx = seg_idx_map[0].cpu().numpy()
            point_cloud = point_cloud.cpu().numpy()[0]
            ground_mask = ground_mask.cpu().numpy()[0]
            nonground_mask = nonground_mask.cpu().numpy()[0]

            # TODO: test only stage1

            total_bit_size = info_bitrate + ground_residual_bitrate + stage_0_rec_bitrate + \
                             real_bitrate_stage_1 / batch_size + real_bitrate_stage_2 / batch_size
            total_compression_ratio = 8.0 * point_cloud.nbytes / total_bit_size
            total_bpp = total_bit_size / (point_cloud.shape[0] * point_cloud.shape[1])
            tb_summary.add_scalar('compress_eval_metrics/total_compression_rate', total_compression_ratio, cur_iter)
            tb_summary.add_scalar('compress_eval_metrics/total_bpp', total_bpp, cur_iter)
            tb_summary.add_scalar('compress_eval_metrics/stage_1_compression_rate',
                                  8.0 * point_cloud.nbytes / (real_bitrate_stage_1 / batch_size), cur_iter)
            tb_summary.add_scalar('compress_len/total_len', total_bit_size / 8, cur_iter)
            # TODO: test only stage1
            tb_summary.add_scalar('compress_eval_metrics/stage_2_compression_rate',
                                  8.0 * point_cloud.nbytes / (real_bitrate_stage_2 / batch_size), cur_iter)

            # print('\n\n', info_bitrate / 8,
            #       ground_residual_bitrate / 8,
            #       stage_0_rec_bitrate / 8,
            #       real_bitrate_stage_1 / batch_size / 8,
            #       real_bitrate_stage_2 / batch_size / 8)
            # for key, value in compressed_data_stage_0.items():
            #     print(key, len(value))
            # IPython.embed()


            # stage_1_acc = 1.0
            # residual_rec_stage_1 = torch.round(residual_rec / stage_1_acc) * stage_1_acc
            # range_image_rec_stage_1 = range_image_rec + residual_rec_stage_1
            # point_cloud_rec_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_1.detach().cpu().numpy()[0],
            #                                          lidar_param.range_image_to_point_cloud_map)
            # compare_point_clouds(point_cloud.cpu().numpy()[0], point_cloud_rec_stage_1, vis_all=True,
            #                      save_path=os.path.join(result_dir, 's1.pcd'), save=True, output=False)

            # # 25000 - 30000 --> residual_unit
            # res, ori, cmp = evaluate_compress_ratio(
            #     point_cloud, range_image, seg_idx_map, residual_rec, ground_model, cluster_xyz, cluster_feats,
            #     0.5)
            # for key, val in ori.items():
            #     print(key, 'original: ', val.nbytes, ', compressed: ', len(cmp[key]))
            # IPython.embed()

            # tb_summary.add_scalar('compress_eval_metrics/bpp', compression_result_dict['bpp'], cur_iter)
            # tb_summary.add_scalar('compress_eval_metrics/bpp_ours_baseline', compression_result_dict['bpp'] / compression_result_dict_baseline['bpp'], cur_iter)
            # tb_summary.add_scalar('compress_eval_metrics/stage_1_compression_ratio', compression_result_dict['total_compression_ratio'], cur_iter)
            # tb_summary.add_scalar('compress_eval_metrics/stage_1_cr_ours_baseline', compression_result_dict['total_compression_ratio'] / compression_result_dict_baseline['total_compression_ratio'], cur_iter)

            # valid_residual_ours = original_data['residual_unit'][np.where(seg_idx > 1)]
            # valid_residual_baseline = original_data_baseline['residual_unit'][np.where(seg_idx > 1)]
            # entropy_ours = calc_shannon_entropy(valid_residual_ours)
            # entropy_baseline = calc_shannon_entropy(valid_residual_baseline)
            # tb_summary.add_scalar('compress_eval_metrics/shannon_entropy_ours_baseline', entropy_ours / entropy_baseline, cur_iter)

            # color_map = COLOR_MAP[seg_idx]
            # save_point_cloud_to_pcd(point_cloud.reshape(-1, 3), os.path.join(result_dir, 'train_plane_map.pcd'),
            #                         color_map.reshape(-1, 3), output=False)

            # save_point_cloud_to_pcd((point_cloud*ground_mask).reshape(-1, 3), os.path.join(result_dir, 'ground.pcd'),
            #                         color_map.reshape(-1, 3), output=False)

            # clustered_pc = pred_dict['clustered_point_cloud'].detach().cpu().numpy()[0] * nonground_mask
            # compare_point_clouds(point_cloud, clustered_pc, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'clustered_pc.pcd'), save=True, output=False)
            # save_point_cloud_to_pcd(clustered_pc.reshape(-1, 3), os.path.join(result_dir, 'c.pcd'), output=False)

            point_cloud_base = nonground_pc_base.detach().cpu().numpy()[0]
            # range_image_rec = np.linalg.norm(clustered_pc, 2, -1, keepdims=True) - range_image_rec * nonground_mask

            # filtered_pc = point_cloud * nonground_mask
            # range_image_base = range_image_base.detach().cpu().numpy()[0]
            point_cloud_rec_stage_1 = point_cloud_rec_stage_1.detach().cpu().numpy()[0]
            point_cloud_rec_stage_2 = point_cloud_rec_stage_2.detach().cpu().numpy()[0]
            range_image_pred_stage_1 = range_image_pred_stage_1.detach().cpu().numpy()[0]
            # TODO: test only stage1
            range_image_pred_stage_2 = range_image_pred_stage_2.detach().cpu().numpy()[0]
            point_cloud_pred_stage_1 = point_cloud_pred_stage_1.detach().cpu().numpy()[0]
            point_cloud_pred_stage_2 = point_cloud_pred_stage_2.detach().cpu().numpy()[0]

            # compare_point_clouds(point_cloud, point_cloud_base, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'pc_rec.pcd'), save=True, output=False)
            # compare_point_clouds(point_cloud, point_cloud_rec_stage_1, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'pc_rec_1.pcd'), save=True, output=False)
            # compare_point_clouds(point_cloud, point_cloud_rec_stage_2, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'pc_rec_2.pcd'), save=True, output=False)
            if i % 500 == 0 and full_eval:
                compare_point_clouds(point_cloud, point_cloud_pred_stage_1, vis_all=True,
                                     save_path=os.path.join(result_dir, 'pc_pred_1.pcd'), save=True, output=False)
                compare_point_clouds(point_cloud, point_cloud_pred_stage_2, vis_all=True,
                                     save_path=os.path.join(result_dir, 'pc_pred_2.pcd'), save=True, output=False)
                compare_point_clouds(point_cloud_rec_stage_1, point_cloud_pred_stage_2, vis_all=True,
                                     save_path=os.path.join(result_dir, 'pc_cmp_2.pcd'), save=True, output=False)
                compare_point_clouds(point_cloud_base, point_cloud_pred_stage_1, vis_all=True,
                                     save_path=os.path.join(result_dir, 'pc_cmp_1.pcd'), save=True, output=False)
            # IPython.embed()

            cd_result_stage_1_pred = calc_chamfer_distance(point_cloud * nonground_mask, point_cloud_pred_stage_1)
            cd_result_stage_2_pred = calc_chamfer_distance(point_cloud * nonground_mask, point_cloud_pred_stage_2)
            cd_result_stage_1_rec = calc_chamfer_distance(point_cloud * nonground_mask, point_cloud_rec_stage_1)
            cd_result_stage_2_rec = calc_chamfer_distance(point_cloud * nonground_mask, point_cloud_rec_stage_2)
            tb_summary.add_scalar('d_chamfer_distance/stage_1_pred_cd_mean', cd_result_stage_1_pred['mean'], cur_iter)
            tb_summary.add_scalar('d_chamfer_distance/stage_2_pred_cd_mean', cd_result_stage_2_pred['mean'], cur_iter)
            tb_summary.add_scalar('d_chamfer_distance/stage_1_rec_cd_mean', cd_result_stage_1_rec['mean'], cur_iter)
            tb_summary.add_scalar('d_chamfer_distance/stage_2_rec_cd_mean', cd_result_stage_2_rec['mean'], cur_iter)

            # range_image = range_image.detach().cpu().numpy()[0]
            # range_image_rec_stage_1 = range_image_rec_stage_1.detach().cpu().numpy()[0]
            # error = range_image - range_image_rec_stage_1
            # print('mean max: ', np.abs(error).mean(), np.abs(error).max())
            # range_image_rec_stage_2 = range_image_rec_stage_2.detach().cpu().numpy()[0]
            # error = range_image - range_image_rec_stage_2
            # print('mean max: ', np.abs(error).mean(), np.abs(error).max())
            # IPython.embed()

            # residual_rec = residual_rec.abs().detach().cpu().numpy()[0]# * nonground_mask
            # residual_baseline = residual_base.abs().detach().cpu().numpy()[0]# * nonground_mask
            # point_cloud_residual_rec = PCTransformer.range_image_to_point_cloud(residual_rec / accuracy + 20, lidar_param.range_image_to_point_cloud_map)
            # point_cloud_residual_clt = PCTransformer.range_image_to_point_cloud(residual_baseline / accuracy + 20, lidar_param.range_image_to_point_cloud_map)
            # compare_point_clouds(point_cloud_residual_clt, point_cloud_residual_rec, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'res.pcd'), save=True, output=False)
            # save_point_cloud_to_pcd(point_cloud_residual_rec.reshape(-1, 3), os.path.join(result_dir, 'res_only.pcd'),
            #                         color_map.reshape(-1, 3), output=False)
            # save_point_cloud_to_pcd(point_cloud_residual_clt.reshape(-1, 3), os.path.join(result_dir, 'res_only_baseline.pcd'),
            #                         color_map.reshape(-1, 3), output=False)

            # print('\nours: ')
            # print('Residual data:')
            # print('plane residual mean: %.2f, plane num: %d, cluster residual mean: %.2f, cluster num: %d\n' %
            #       (np.mean(np.abs(original_data['plane_residual'])), original_data['plane_residual'].shape[0],
            #        np.mean(np.abs(original_data['cluster_residual'])), original_data['cluster_residual'].shape[0]))
            # for key, val in original_data.items():
            #     print(key, 'original: ', val.nbytes, ', compressed: ', len(compressed_data[key]))
            #
            # bit_stream = compressed_data['residual_unit'] + \
            #              compressed_data['contour_map'] + \
            #              compressed_data['idx_sequence'] + \
            #              compressed_data['plane_param'] + \
            #              compressed_data['cluster_xyz'] + \
            #              compressed_data['cluster_feats']
            # print('\nCompression ratio:')
            # cr = point_cloud.nbytes / len(bit_stream)
            # print('CR: %.2f, bpp: %.2f, (residual %.1f%%, contour_map %.1f%%, idx_sequence %.1f%%, cluster_xyz %.1f%%, cluster_feats %.1f%%)'
            #       % (
            #       cr, (len(bit_stream) * 8) / point_cloud.shape[0] / point_cloud.shape[1],
            #       len(compressed_data['residual_unit']) / len(bit_stream) * 100
            #       , len(compressed_data['contour_map']) / len(bit_stream) * 100
            #       , len(compressed_data['idx_sequence']) / len(bit_stream) * 100
            #       , len(compressed_data['cluster_xyz']) / len(bit_stream) * 100,
            #       len(compressed_data['cluster_feats']) / len(bit_stream) * 100)
            # )
            #
            # range_image_reconstructed = range_image_rec.detach().cpu().numpy()[0, :, :, 0] + original_data['residual_unit'] * accuracy
            # dif = range_image_reconstructed - range_image.cpu().numpy()[0, :, :, 0]
            # print('our method max error: %.4f, mean error: %.4f.' % (np.max(np.abs(dif)), np.mean(np.abs(dif))))
            #
            # print('\nbaseline: ')
            # print('plane residual mean: %.2f, plane num: %d, cluster residual mean: %.2f, cluster num: %d\n' %
            #       (np.mean(np.abs(original_data_baseline['plane_residual'])), original_data_baseline['plane_residual'].shape[0],
            #        np.mean(np.abs(original_data_baseline['cluster_residual'])), original_data_baseline['cluster_residual'].shape[0]))
            # for key, val in original_data_baseline.items():
            #     print(key, 'original: ', val.nbytes, ', compressed: ', len(compressed_data_baseline[key]))
            #
            # from utils.compress_utils import BasicCompressor
            # basic_compressor = BasicCompressor(method_name='bzip2')
            # # basic_compressor = BasicCompressor(method_name='lz4')
            # range_image = range_image.cpu().numpy()[0]
            # resolution_0 = 2.0
            # range_image_quan = (np.rint(range_image / resolution_0) * resolution_0).astype(np.int8)
            # len(basic_compressor.compress(range_image_quan))
            # res_quan = range_image - range_image_quan
            # IPython.embed()
            # r = original_stage_2_data['residual_unit']
            # l = len(basic_compressor.compress(r.astype(np.int8)))
            # rr = np.abs(r)
            # rr_pc = PCTransformer.range_image_to_point_cloud(np.expand_dims(rr, -1), lidar_param.range_image_to_point_cloud_map)
            # save_point_cloud_to_pcd(rr_pc.reshape(-1, 3),
            #                         os.path.join(result_dir, 'a.pcd'),
            #                         color_map.reshape(-1, 3), output=False)
            #
            #
            # r = original_data['residual_unit'][np.where(seg_idx > 1)]
            #
            #
            #

            # point_cloud_rec_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_1.detach().cpu().numpy()[0],
            #                                                                    lidar_param.range_image_to_point_cloud_map)
            # point_cloud_rec_stage_2 = PCTransformer.range_image_to_point_cloud(range_image_rec_stage_2.detach().cpu().numpy()[0],
            #                                                                    lidar_param.range_image_to_point_cloud_map)
            # compare_point_clouds(point_cloud, point_cloud_rec_stage_1, vis_all=True,
            #                      save_path=os.path.join(result_dir, 's1.pcd'), save=True, output=False)
            # compare_point_clouds(point_cloud, point_cloud_rec_stage_2, vis_all=True,
            #                      save_path=os.path.join(result_dir, 's2.pcd'), save=True, output=False)
            # compare_point_clouds(point_cloud_rec_stage_1, point_cloud_rec_stage_2, vis_all=True,
            #                      save_path=os.path.join(result_dir, 's12.pcd'), save=True, output=False)
            # IPython.embed()

            #
            # print('mean_reconstruct_stage_1', residual_rec.abs().mean().item())
            # print('mean_baseline_stage_1', residual_base.abs().mean().item())
            # print('len_reconstruct_stage_1', len(compressed_data['residual_unit']))
            # print('len_baseline_stage_1', len(compressed_data_baseline['residual_unit']))
            # IPython.embed()
            # save_point_cloud_to_pcd((point_cloud * ground_mask).reshape(-1, 3), os.path.join(result_dir, 'a.pcd'),
            #                         output=False)
            # save_point_cloud_to_pcd((point_cloud * (1 - ground_mask)).reshape(-1, 3), os.path.join(result_dir, 'b.pcd'),
            #                         output=False)
            # IPython.embed()

            # range_image_reconstruct_stage_0 = pred_dict['reconstructed_range_image_stage_0'].detach().cpu().numpy()[0]
            # point_cloud_reconstruct_stage_0 = PCTransformer.range_image_to_point_cloud(range_image_reconstruct_stage_0, lidar_param)[zero_idx]
            # compare_point_clouds(point_cloud, point_cloud_reconstruct_stage_0, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'train_reconstruct_stage_0.pcd'), save=True, output=False)
            #
            # range_image_reconstruct_stage_1 = pred_dict['reconstructed_range_image_stage_1'].detach().cpu().numpy()[0]
            # # range_image_reconstruct_stage_1 += range_image_reconstruct_stage_0
            # point_cloud_reconstruct_stage_1 = PCTransformer.range_image_to_point_cloud(range_image_reconstruct_stage_1,
            #                                                                            lidar_param)[zero_idx]
            # compare_point_clouds(point_cloud, point_cloud_reconstruct_stage_1, vis_all=True,
            #                      save_path=os.path.join(result_dir, 'train_reconstruct_stage_1.pcd'), save=True,
            #                      output=False)

        # pbar.set_postfix(train_cr=train_cr, val_cr=val_cr)
        pbar.set_postfix(loss=loss_G.item())
        if train:
            pbar.set_description(
                'Epoch %d, iter %d--Train' % (cur_iter // (args.batch_size * len(data_loader)), cur_iter))
        else:
            pbar.set_description('Val')
        pbar.update()
        pbar.refresh()
    return cur_iter


if __name__ == '__main__':
    if not args.test:
        train()
    else:
        test()
