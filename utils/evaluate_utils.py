import IPython, os
from pathlib import Path
import copy
from utils.compress_utils import BasicCompressor, compress_plane_idx_map, compress_point_cloud, calc_residual
import numpy as np
from utils.ChamferDistancePytorch.chamfer3D import dist_chamfer_3D
from utils.ChamferDistancePytorch import fscore
import torch
import numpy as np
# from chamferdist import ChamferDistance


def sys_size(data):
    return sys.getsizeof(data)


def bit_size(data):
    return len(data)


def np_size(data):
    return data.nbytes


def calc_shannon_entropy(data):
    if len(data) <= 1:
        return 0
    value, counts = np.unique(data, return_counts=True)
    probs = counts / len(data)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # base = e if base is None else base
    for i in probs:
        ent -= i * np.log(i)
    return ent


def evaluate_compress_ratio(point_cloud, range_image, cluster_idx, residual, plane_param, cluster_xyz, cluster_feats, accuracy=0.02, compress_method=None, compressor_yaml=None, full=False):
    if plane_param is not None:
        plane_param = plane_param.detach().cpu().numpy()[0]
    cluster_xyz = cluster_xyz.detach().cpu().numpy()[0]
    if cluster_feats is not None:
        cluster_feats = cluster_feats.detach().cpu().numpy()[0]
    point_cloud = point_cloud.cpu().numpy()[0]
    range_image = range_image.cpu().numpy()[0]
    cluster_idx = cluster_idx.detach().cpu().numpy()[0].reshape(point_cloud.shape[0], point_cloud.shape[1])
    if residual is not None:
        residual = residual.detach().cpu().numpy()[0].reshape(point_cloud.shape[0], point_cloud.shape[1])
    # if residual.is_cuda:
    #     residual = residual.detach().cpu().numpy()[0].reshape(point_cloud.shape[0], point_cloud.shape[1])
    # if error_factor.is_cuda:
    #     error_factor = error_factor.detach().cpu().numpy()[0].reshape(point_cloud.shape[0], point_cloud.shape[1])

    # accuracy = 0.02 / error_factor
    # residual_unit = residual / accuracy

    # pick large error points to single save. (save as int16)
    # make sure the error < 0.02

    if compress_method is None and compressor_yaml is None:
        basic_compressor = BasicCompressor(method_name='bzip2')
    else:
        basic_compressor = BasicCompressor(method_name=compress_method, compressor_yaml=compressor_yaml)

    # residual = calc_residual(range_image, plane_param, cluster_param, cluster_idx, lidar_param)
    original_data, compressed_data = compress_point_cloud(point_cloud, range_image, plane_param, cluster_xyz, cluster_feats, cluster_idx, residual, accuracy,
                         basic_compressor)

    compression_ratio = copy.deepcopy(compressed_data)
    total_bit_size = 0
    for key, val in compressed_data.items():
        if key in ['point_cloud', 'range_image']:
            del compression_ratio[key]
            continue
        if key in ['residual_unit', 'contour_map', 'idx_sequence', 'cluster_xyz', 'cluster_feats']:
            total_bit_size += 8.0 * len(val)
        compression_ratio[key] = np_size(point_cloud) / len(val)
    compression_ratio['total_compression_ratio'] = 8.0 * np_size(point_cloud) / total_bit_size
    compression_ratio['bpp'] = total_bit_size / (point_cloud.shape[0] * point_cloud.shape[1])
    compression_ratio['total_bitrate'] = total_bit_size

    return compression_ratio, original_data, compressed_data


def calc_chamfer_distance(points1, points2):
    pc_1 = points1[np.where(np.sum(points1, -1) != 0)]
    pc_2 = points2[np.where(np.sum(points2, -1) != 0)]
    pc_1 = torch.from_numpy(pc_1).unsqueeze(0).cuda().float()  # 98512 * 3
    pc_2 = torch.from_numpy(pc_2).unsqueeze(0).cuda().float()  # 98512 * 3
    # chamfer_distance = dist_chamfer(pc_1, pc_2)
    chamLoss = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = chamLoss(pc_1, pc_2)
    f_score, precision, recall = fscore.fscore(dist1, dist2)
    # print('dist1: %.4f, dist2: %.4f' %
    # (torch.sqrt(dist1).mean().cpu().numpy(), torch.sqrt(dist2).mean().cpu().numpy()))
    # cham_dist1 = torch.sqrt(dist1.sum()).cpu().item() / pc_1.shape[1]
    # cham_dist2 = torch.sqrt(dist2.sum()).cpu().item() / pc_2.shape[1]
    cham_dist1 = torch.sqrt(dist1).mean().item()
    cham_dist2 = torch.sqrt(dist2).mean().item()

    result = {
        'max': max(cham_dist1, cham_dist2),
        'mean': (cham_dist1 + cham_dist2) / 2,
        'sum': cham_dist1 + cham_dist2,
        'cd1': cham_dist1,
        'cd2': cham_dist2,
        'f_score': f_score.item(),
        'precision': precision.item(),
        'recall': recall.item(),
        'chamfer_dist_info': {
            'dist1': dist1.cpu().numpy()[0],
            'dist2': dist2.cpu().numpy()[0],
            'idx1': idx1.cpu().numpy()[0],
            'idx2': idx2.cpu().numpy()[0],
        }
    }

    # a = torch.from_numpy(np.array([[[1, 1, 1], [1, 1, 2]]])).cuda().float()
    # b = torch.from_numpy(np.array([[[1, 1, 1], [1, 1, 3], [1, 1, -1]]])).cuda().float()
    # chamferDist = ChamferDistance()
    # dist_forward = chamferDist(b, a)
    # dist_backward = chamferDist(a, b)
    # chamferDist = ChamferDistance()
    # dist_forward = chamferDist(pc_1, pc_2) / pc_1.shape[1]
    # dist_backward = chamferDist(pc_2, pc_1) / pc_2.shape[1]
    return result
