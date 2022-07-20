import torch
import torch.nn as nn
from utils.ops.pointnet2.pointnet2_batch import pointnet2_modules


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels):
        '''
        model_cfg
            SA_CONFIG
                NPOINTS
                RADIUS
                NSAMPLE
                MLPS
            FP_CONFIG
                MLPS
        '''
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                    relative=self.model_cfg.SA_CONFIG.USE_REL[k]
                    # relative = self.model_cfg.SA_CONFIG.get('USE_REL', False)
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_CONFIG.MLPS.__len__()):
            pre_channel = self.model_cfg.FP_CONFIG.MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_CONFIG.MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_CONFIG.MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_CONFIG.MLPS[0][-1]

    def forward(self, xyz, features):
        """
        Args:
            xyz: B x N x 3
            features: B x N x C
        Returns:
            features: B x N x C
        """
        # features = features.permute(0, 2, 1).contiguous() if features is not None else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        point_features = l_features[0]  #.permute(0, 2, 1).contiguous()  # (B, N, C)
        return point_features
