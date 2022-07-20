from utils.ops.pointnet2.pointnet2_batch import pointnet2_utils
from utils.ops.pointnet2.pointnet2_batch import pointnet2_modules
import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython
import numpy as np

class RangeAttention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.1, alpha=0.2, radius=0.5, nsample=10):
        super(RangeAttention, self).__init__()
        self.alpha = alpha
        # self.a = nn.Parameter(torch.zeros(size=(out_channels + 4, out_channels)))
        self.a = nn.Parameter(torch.zeros(size=(4, out_channels))) # only pos
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.dropout = dropout
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.bn = nn.BatchNorm2d(out_channels)
        # self.grouper = pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=False) 

    def forward(self, range_image, point_cloud, feats, kernel_size=7):
        '''
        Input:
            feats: [b, c, h, w]
            nsample = kernel size - 1
        Return:
            feats_with_attention: [b, h, w, c]
        '''
        # conv first
        feats = self.bn(self.conv(feats)).permute(0, 2, 3, 1)
        
        # attention
        b, h, w, c = feats.shape
        
        pos = torch.cat((range_image, point_cloud), -1)
        grouped_pos = self.group(pos, kernel_size)  # [B, H, W, nsample, 4]
        delta_pos = pos.view(b, h, w, 1, 4) - grouped_pos
        grouped_feats = self.group(feats, kernel_size)  # [B, H, W, nsample, C]
        # # pos with feats
        # delta_feats = feats.view(b, h, w, 1, c) - grouped_feats
        # delta_p_concat_h = torch.cat((delta_pos, delta_feats), -1)  # [B, H, W, nsample, C + 4]
        # e = self.leakyrelu(torch.matmul(delta_p_concat_h, self.a)) # [B, H, W, nsample, C]
        
        # only pos
        e = self.leakyrelu(torch.matmul(delta_pos, self.a)) # [B, H, W, nsample, C]
        
        attention = F.softmax(e, dim=3) # [B, H, W, nsample, C]
        attention = F.dropout(attention, self.dropout, training=self.training)
        att_feats = torch.sum(torch.mul(attention, grouped_feats), dim = 3) # [B, H, W, C]
        return att_feats.permute(0, 3, 1, 2)
    
    def group(self, feats, kernel_size):
        b, h, w, c = feats.shape
        pad_feats = F.pad(feats, (0, 0, kernel_size//2, kernel_size//2), "constant", 0)
        group_feats = []
        for i in range(kernel_size):
            if i == 0:
                continue
            k_feats = pad_feats[:, :, i:i + w, :].view(b, h, w, 1, c)
            group_feats.append(k_feats)
        group_feats = torch.cat(group_feats, 3)
        return group_feats
