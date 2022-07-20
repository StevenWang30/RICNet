import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython
import numpy as np
# from models.loss import Loss
# from models.pointseg_net import PointSegNet
from models.squeezeseg_net import Fire, SqueezeRefine
# from models.squeeze_refine import SqueezeRefine
from models.pointnet2_backbone import PointNet2MSG
import MinkowskiEngine as ME
from models.minkunet import MinkUNet14, MinkUNet34C, MinkUNet101
import time
from dataset import PCTransformer
from easydict import EasyDict as edict
from models.range_attention_net import RangeAttention


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, backbone='MinkUNet14', voxel_size=0.2, use_attention=True):  # or 2d  or 2d-easy or pointnet or mink
        super(FeatureExtractor, self).__init__()
        self.method = backbone
        
        if backbone == '2d':
            self.extractor = SqueezeRefine(in_channels=in_channels, out_channels=out_channels)
            mid_channels = self.extractor.feats_channels
            
        elif backbone == '2d-easy':
            self.extractor = nn.Sequential(
                nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
            )
            mid_channels = 256
            
        elif backbone == 'pointnet':
            model_cfg = edict()
            model_cfg.SA_CONFIG = edict()
            model_cfg.SA_CONFIG.USE_XYZ = False
            # model_cfg.SA_CONFIG.USE_REL = True
            model_cfg.SA_CONFIG.USE_REL = [True, False, False]
            model_cfg.SA_CONFIG.NPOINTS = [4096, 1024, 256]
            model_cfg.SA_CONFIG.NSAMPLE = [[16, 32], [16, 32], [16, 32]]
            model_cfg.SA_CONFIG.RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0]]
            model_cfg.SA_CONFIG.MLPS = [[[16, 16, 32], [32, 32, 64]],
                                        [[64, 64, 128], [64, 96, 128]],
                                        [[128, 196, 256], [128, 196, 256]]]

            # model_cfg.SA_CONFIG.NPOINTS = [1024, 512, 256]
            # model_cfg.SA_CONFIG.NSAMPLE = [[8, 16], [8, 16], [8, 16]]
            # model_cfg.SA_CONFIG.RADIUS = [[0.3, 0.6], [0.6, 1.0], [1.5, 2.0]]
            # model_cfg.SA_CONFIG.MLPS = [[[16, 16, 32], [32, 32, 64]],
            #                             [[64, 64, 128], [64, 96, 128]],
            #                             [[128, 196, 256], [128, 196, 256]]]
            model_cfg.FP_CONFIG = edict()
            model_cfg.FP_CONFIG.MLPS = [[64, 64], [128, 128], [256, 256]]
            self.extractor = PointNet2MSG(model_cfg, input_channels=in_channels + 3)
            mid_channels = 64
            
        elif 'Mink' in backbone:
            self.voxel_size = voxel_size
            if backbone == 'MinkUNet14':
                self.extractor = MinkUNet14(in_channels, out_channels)
            elif backbone == 'MinkUNet101':
                self.extractor = MinkUNet101(in_channels, out_channels)
            elif backbone == 'MinkUNet34':
                self.extractor = MinkUNet34C(in_channels, out_channels)
            mid_channels = self.extractor.feats_channels
        else:
            NotImplementedError
        
        if use_attention:
            self.attention_layer1 = RangeAttention(mid_channels, 128, kernel_size=3)
            self.attention_layer2 = RangeAttention(128, 64, kernel_size=3)
        else:
            self.out_conv1 = nn.Sequential(
                nn.Conv2d(mid_channels, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
            self.out_conv2 = nn.Sequential(
                nn.Conv2d(128, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
            )
        
        self.use_attention = use_attention
        self.mid_channels = mid_channels
        self.out_channels = out_channels

    def forward(self, point_cloud, range_image, feats, skip_feats=None):
        b, h, w, _ = point_cloud.shape
        if self.method == '2d':
            # feats = torch.cat((point_cloud, range_image), -1).permute(0, 3, 1, 2)
            feats = self.extractor(feats, skip_feats)
        elif self.method == '2d-easy':
            feats = self.extractor(feats)
        elif self.method == 'pointnet':
            xyz = point_cloud.view(b, h * w, 3)
            feats = feats.view(b, feats.shape[1], h * w)
            feats = self.extractor(xyz, feats)
            feats = feats.view(b, self.mid_channels, h, w)  # B x H x W x Q_len
        elif 'Mink' in self.method:
            coords = point_cloud.view(b, h * w, 3) / self.voxel_size
            feats = feats.view(b, feats.shape[1], h * w).transpose(1, 2)
            coords, feats = ME.utils.sparse_collate([c for c in coords],
                                                    [f for f in feats])
            in_field = ME.TensorField(
                features=feats,
                coordinates=coords,
                quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE,
                minkowski_algorithm=ME.MinkowskiAlgorithm.SPEED_OPTIMIZED,
                device=feats.device,
            )
            # Convert to a sparse tensor
            sinput = in_field.sparse()
            # Output sparse tensor
            feats, prob = self.extractor(sinput)
            # get the prediction on the input tensor field
            
            # prob = prob.slice(in_field)
            # prob = prob.F.view(b, h, w, self.Q_len)
            
            feats = feats.slice(in_field)
            feats = feats.F.view(b, h, w, feats.shape[-1]).permute(0, 3, 1, 2)
        
        if self.use_attention:
            # attention layers            
            feats = self.attention_layer1(range_image, point_cloud, feats, kernel_size=3)
            feats = self.attention_layer2(range_image, point_cloud, feats, kernel_size=3)
        else:
            # original conv layers
            feats = self.out_conv2(self.out_conv1(feats))
        
        return feats


class RoundNoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, g):
        return g


class RICNet(nn.Module):
    def __init__(self, lidar_param, O_len=10, stage_1_bb='MinkUNet14', stage_2_bb='MinkUNet14',
                 stage_1_voxel_size=0.3, stage_2_voxel_size=0.2, use_attention=True):  # , bilinear=True):
        super(RICNet, self).__init__()
        self.transform_map = torch.from_numpy(lidar_param.range_image_to_point_cloud_map).float().cuda()

        self.O_len = O_len
        input_channels = 1
        output_channels = 64
        self.stage_1_feature_extractor = FeatureExtractor(input_channels, output_channels, backbone=stage_1_bb, voxel_size=stage_1_voxel_size, use_attention=use_attention)
        self.occupancy_head = nn.Conv2d(output_channels, O_len, kernel_size=1)
        self.stage_2_feature_extractor = FeatureExtractor(input_channels, output_channels, backbone=stage_2_bb, voxel_size=stage_2_voxel_size, use_attention=use_attention)
        self.refinement_head = nn.Conv2d(output_channels, 1, kernel_size=1)
        
        self.quantize_func = RoundNoGradient()

    def quantize(self, data):
        return self.quantize_func.apply(data)

    def RICNet_stage_1(self, point_cloud, range_image, feats, acc):
        feats = self.stage_1_feature_extractor(point_cloud, range_image, feats)
        occ_prob = self.occupancy_head(feats)
        occ_prob = F.softmax(occ_prob.permute(0, 2, 3, 1), -1)
        
        # reconstruct point cloud
        # use expected value to predict regression to reconstruct quantized residual and range image
        res_template = torch.arange(-(self.O_len // 2), (self.O_len // 2) + 1).cuda() * acc
        expected_residual = torch.sum(res_template.view(1, 1, 1, self.O_len) * occ_prob, -1, keepdim=True)
        range_image_rec = range_image - expected_residual
        return occ_prob, range_image_rec
    
    def RICNet_stage_2(self, point_cloud, range_image, feats, acc):
        feats = self.stage_2_feature_extractor(point_cloud, range_image, feats)
        residual = self.refinement_head(feats)
        
        # with Sigmoid
        residual = (torch.sigmoid(residual.permute(0, 2, 3, 1)) - 0.5) * acc
        
        # # without sigmoid 
        # residual = residual.permute(0, 2, 3, 1)
        
        range_image_rec = range_image - residual
        
        return range_image_rec
        

def calc_plane_residual_vertical(point_cloud, plane_param):
    '''
        plane_param: 4 (ax + by + cz + d)
        point to plane: d = |ax + by + cz + d| / sqrt(a**2 + b**2 + c**2)
        output: residual is same as distance   H x W
    '''
    plane_param = plane_param.unsqueeze(1).unsqueeze(2)
    residual = torch.abs(torch.sum(point_cloud * plane_param[..., :3], -1) + plane_param[..., 3]) / \
               torch.norm(plane_param[..., :3], 2, -1)
    return residual.unsqueeze(-1)