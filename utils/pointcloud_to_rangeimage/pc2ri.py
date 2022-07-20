import sys
sys.path.append("..")
from pointcloud_to_rangeimage import pointcloud_to_rangeimage
import glob
import numpy as np
import IPython
import open3d as o3d
import os

lidar_angle_xy_range_ = 360
max_lidar_angle_z_ = 2.5
min_lidar_angle_z_ = -24.9
range_x_ = 64
range_y_ = 2000
nearest_bound_ = 0.0
furthest_bound_ = 100
if_show_ground_ = True

# files = glob.glob('/data/rangeimage_prediction/pointcloud_txt_file/*/*.txt')
files = glob.glob('/data/KITTI_rawdata/2011_09_26_drive_0011_extract/velodyne_points/data/*.txt')
save_dir = '/data/rangeimage_prediction_64E_32E_16E/64E/IROS2021_data/2011_09_26_drive_0011_extract/range_image'
os.makedirs(save_dir, exist_ok=True)
print('create directory: ', save_dir)

files.sort()
for f in files:
    pointcloud = np.loadtxt(f)
    pointcloud = pointcloud[..., :3]
    print(f)
    ri = pointcloud_to_rangeimage(pointcloud, lidar_angle_xy_range_, max_lidar_angle_z_, min_lidar_angle_z_,
                             range_x_, range_y_)

    range_image_save_path = os.path.join(save_dir, f.split('/')[-1])
    np.savetxt(range_image_save_path, ri, fmt='%.5f')