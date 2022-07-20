# import open3d as o3d
import imageio
import glob
import os
import torch
from pathlib import Path
import numpy as np
# import pyransac3d as pyrsc
import open3d as o3d
import IPython


def save_range_image_to_png(range_image, save_path):
    imageio.imwrite(save_path, range_image)


def save_checkpoint(model, ckpt_dir, ckpt_name, cur_iter, cur_epoch, max_ckpt_save_num=None, single=False):
    # clean ckpt file to keep max_ckpt_save_num
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if not single:
        ckpt_list = glob.glob(str(ckpt_dir) + '/checkpoint_' + ckpt_name + '_epoch_*.pth')
        ckpt_list.sort(key=os.path.getmtime)
        if ckpt_list.__len__() >= max_ckpt_save_num:
            for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                os.remove(ckpt_list[cur_file_idx])
        ckpt_name = ckpt_dir / ('checkpoint_' + ckpt_name + '_epoch_' + str(cur_epoch + 1) + '.pth')
    else:
        ckpt_name = ckpt_dir / (ckpt_name + '.pth')
    checkpoint = {
        'model': model.state_dict(),
        'cur_iter': cur_iter,
        'cur_epoch': cur_epoch,
    }
    torch.save(checkpoint, ckpt_name)
    if not single:
        print('save model weight to ' + str(ckpt_name))


def load_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path)
    cur_iter = checkpoint['cur_iter']
    cur_epoch = checkpoint['cur_epoch']
    model = checkpoint['model']
    return model, cur_iter, cur_epoch


def code_backup(save_name, save_path):
    save_content = [
        'train.py',
        'dataset.py',
        'utils/*.py',
        'models/*.py',
    ]
    save_dir = os.path.join(save_path, save_name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for file_name in save_content:
        if '*' not in file_name:
            os.system('cp ' + file_name + ' ' + os.path.join(save_dir, file_name))
        else:
            for file in glob.glob(file_name):
                save_file_path = os.path.join(save_dir, file)
                Path(save_file_path).parent.mkdir(parents=True, exist_ok=True)
                os.system('cp ' + file + ' ' + save_file_path)


# def torch_dict_to_idx_numpy(torch_dict, plane_idx):
#     idx = plane_idx.detach().cpu().numpy()
#     numpy_dict = {}
#     for key, val in torch_dict.items():
#         val_np = val.detach().cpu().numpy()
#         numpy_dict[key] = val_np[idx]
#     return numpy_dict

def plane_segmentation(point_cloud, threshold=0.1, ransac_n=10, num_iterations=100):
    # o3d
    points_o3d = o3d.geometry.PointCloud()
    points_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    coefficients, indices = points_o3d.segment_plane(distance_threshold=threshold,
                                                     ransac_n=ransac_n,
                                                     num_iterations=num_iterations)
    # [a, b, c, d] = plane_model

    # # pcl segmentation fault
    # import pcl
    # cloud = pcl.PointCloud()
    # cloud.from_array(point_cloud.astype(np.float32))
    # seg = cloud.make_segmenter_normals(ksearch=50)
    # seg.set_optimize_coefficients(True)
    # seg.set_model_type(pcl.SACMODEL_NORMAL_PLANE)
    # seg.set_method_type(pcl.SAC_RANSAC)
    # seg.set_distance_threshold(0.01)
    # seg.set_normal_distance_weight(0.01)
    # seg.set_max_iterations(100)
    # indices, coefficients = seg.segment()

    # # pyransac3d
    # import pyransac3d as pyrsc
    # plane1 = pyrsc.Plane()
    # coefficients, indices = plane1.fit(point_cloud, threshold)
    # # print('plane coefficients: ', coefficients)
    return indices, np.array(coefficients)


def calc_plane_residual_depth(range_image, plane_param, transform_map):
    '''
        plane_param: 4 (ax + by + cz + d)
        transform_map: [A, B, C] for {x, y, z} --> x = A * depth, y = B * depth, z = C * depth
        center to point r: range_image depth
        center to plane range r': aAr' + bBr' + cCr' + d = 0 --> r' = -d / (aA + bB + cC)
        residual: delta_r = |r - r'|
        use square of the residual as the loss
        output: residual is same as distance
    '''
    range_image_reshape = np.reshape(range_image, -1)
    r_point = range_image_reshape
    transform_map = np.reshape(transform_map, (-1, 3))
    r_plane = -np.expand_dims(plane_param[3], 0) / \
              np.sum(np.expand_dims(plane_param[:3], 0) * transform_map, -1)
    residual = r_point - r_plane
    return residual.reshape(range_image.shape)


def calc_plane_residual_vertical(point_cloud, plane_param):
    '''
        plane_param: 4 (ax + by + cz + d)
        point to plane: d = |ax + by + cz + d| / sqrt(a**2 + b**2 + c**2)
        output: residual is same as distance
    '''
    residual = np.abs(np.sum(point_cloud * plane_param[:3], -1) + plane_param[3]) / np.linalg.norm(plane_param[:3])
    return residual

