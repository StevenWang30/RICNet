import numpy as np
import open3d as o3d
import torch.utils.data as torch_data
from collections import defaultdict
import IPython
from easydict import EasyDict
import yaml
import math
import random
import torch
import time
from utils.utils import plane_segmentation, calc_plane_residual_depth, calc_plane_residual_vertical


class LidarParam():
    def __init__(self, lidar_yaml=None):
        with open(lidar_yaml, 'r') as f:
            try:
                config = yaml.load(f, Loader=yaml.FullLoader)
            except:
                config = yaml.load(f)
        lidar_config = EasyDict(config)
        self.xy_angle_range = lidar_config.XY_ANGLE_RANGE * (np.pi / 180)
        self.z_angle_max = lidar_config.Z_ANGLE_MAX * (np.pi / 180)
        self.z_angle_min = lidar_config.Z_ANGLE_MIN * (np.pi / 180)
        self.nearest_range = lidar_config.NEAREST_RANGE
        self.furthest_range = lidar_config.FURTHEST_RANGE
        self.range_image_height = lidar_config.RANGE_IMAGE_HEIGHT
        self.range_image_width = lidar_config.RANGE_IMAGE_WIDTH
        self.channels = lidar_config.CHANNEL

        self.range_image_to_point_cloud_map = self.create_ri_to_pc_map()

    def create_ri_to_pc_map(self):
        width = self.range_image_width
        height = self.range_image_height
        # max_range = self.furthest_range
        # min_range = self.nearest_range
        min_angle = self.z_angle_min
        max_angle = self.z_angle_max
        d_altitude = (self.z_angle_max - self.z_angle_min) / height
        d_azimuth = self.xy_angle_range / width

        print('create transform map...')
        transform_map = np.zeros((height, width, 3))
        for r in range(height):
            for c in range(width):
                # depth = range_image[b, r, c, 0] * (max_range - min_range) + min_range;
                altitude = r * d_altitude + min_angle
                # altitude = max_angle - r * d_altitude
                azimuth = c * d_azimuth
                # if not isinstance(depth, np.ndarray):
                transform_map[r, c, 0] = math.cos(altitude) * math.cos(azimuth)  # * depth
                transform_map[r, c, 1] = math.cos(altitude) * math.sin(azimuth)  # * depth
                transform_map[r, c, 2] = math.sin(altitude)  # * depth

        return transform_map


class Dataset(torch_data.Dataset):
    def __init__(self, datalist=None, lidar_param=None, remove_outlier=False, plane_seg=True, random_sample=None, limit_sample=None, aug=False):
        # assert train is True or test is True or val is True
        self.data_list = []
        # print('start load data list from ', datalist)
        if datalist:
            for line in open(datalist, "r"):
                self.data_list.append(line.strip())

        if random_sample is not None:
            self.data_list = random.sample(self.data_list, min(len(self.data_list), random_sample))
        else:
            if limit_sample is not None:
                self.data_list = self.data_list[:min(limit_sample, len(self.data_list))]
        # lidar param
        self.use_radius_outlier_removal = remove_outlier
        self.plane_segmentation = plane_seg
        if lidar_param:
            self.lidar_param = lidar_param
            self.transform_map = lidar_param.range_image_to_point_cloud_map

        self.aug = aug
        # print('dataset initialize finished.')

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # s = time.time()
        file = self.data_list[index]
        point_cloud = self.load_pc(file)

        if self.use_radius_outlier_removal:
            # too slow. plz preprocess data.
            points_o3d = o3d.geometry.PointCloud()
            points_o3d.points = o3d.utility.Vector3dVector(point_cloud)
            cloud_filtered, index = points_o3d.remove_radius_outlier(nb_points=3, radius=1)
            point_cloud = np.asarray(cloud_filtered.points)

        range_image = PCTransformer.pointcloud_to_rangeimage(point_cloud, lidar_param=self.lidar_param)
        range_image = np.expand_dims(range_image, -1)

        if self.aug:
            range_image = self.augment(range_image)
        point_cloud = PCTransformer.range_image_to_point_cloud(range_image, self.transform_map)

        if self.plane_segmentation:
            pc_filter = point_cloud[np.where(point_cloud[..., 2] < -1.0)]
            if pc_filter.shape[0] < 5000:
                pc_filter = point_cloud[np.where(point_cloud[..., 2] < 0)]
            else:
                random_idx = np.random.choice(pc_filter.shape[0], 5000, replace=False)
                pc_filter = pc_filter[random_idx]
            t = time.time()
            indices, ground_model = plane_segmentation(pc_filter)
            # print('ransac: ', time.time() - t)

            # use depth residual
            # depth_dif = calc_plane_residual_depth(range_image, ground_model, self.transform_map)
            # ground_mask[np.where(np.abs(depth_dif) < 0.1)] = 1

            # use vertical distance
            depth_dif = calc_plane_residual_vertical(point_cloud, ground_model)
            plane_idx = np.where(np.abs(depth_dif) < 0.4)
            zero_idx = np.where(range_image == 0)
            # ground_mask = np.zeros_like(range_image)
            # ground_mask[plane_idx] = 1
            # ground_mask[zero_idx] = 0
            # nonground_mask = np.ones_like(range_image)
            # nonground_mask[plane_idx] = 0
            # nonground_mask[zero_idx] = 0
            # print('time cost for all: ', time.time() - s)
            return point_cloud, range_image, ground_model, file
        else:
            return point_cloud, range_image, file

    def get_test_data(self, index):
        file_name = self.data_list[index]
        if self.plane_segmentation:
            point_cloud, range_image, ground_model = self.__getitem__(index)
            return point_cloud, range_image, ground_model, file_name
        else:
            point_cloud, range_image = self.__getitem__(index)
            return point_cloud, range_image, file_name
    
    def load_pc(self, file):
        type = file.split('.')[-1]
        if type == 'txt':
            point_cloud = np.loadtxt(file)
        elif type == 'bin':
            point_cloud = np.fromfile(file, dtype=np.float32)
            point_cloud = point_cloud.reshape((-1, 4))
            point_cloud = point_cloud[:, :3]
        elif type == 'npy' or type == 'npz':
            point_cloud = np.load(file, allow_pickle=True)
        else:
            assert False, 'File type not correct.'
        return point_cloud

    def augment(self, range_image):
        ran_f = 1 + (2 * np.random.rand() - 1) * 0.2
        range_image *= ran_f
        return range_image


class PCTransformer:
    @staticmethod
    def calc_points_vertical_angle(points):
        return np.arctan2(points[:, 2], np.linalg.norm(points[:, :2], 2, -1))

    @staticmethod
    def calc_points_horizon_angle(points):
        return np.arctan2(points[:, 1], points[:, 0]) % (2 * np.pi)

    @staticmethod
    def pointcloud_to_rangeimage(pointcloud, lidar_param):
        range_image = np.zeros((lidar_param.range_image_height, lidar_param.range_image_width), dtype=np.float32)
        xy_angle = PCTransformer.calc_points_horizon_angle(pointcloud)
        col = (xy_angle / lidar_param.xy_angle_range * lidar_param.range_image_width).astype(np.int32)
        z_angle = PCTransformer.calc_points_vertical_angle(pointcloud)
        z_angle[np.where(z_angle <= lidar_param.z_angle_min)] = lidar_param.z_angle_min + 1e-6
        z_angle[np.where(z_angle >= lidar_param.z_angle_max)] = lidar_param.z_angle_max - 1e-6
        row = np.rint((z_angle - lidar_param.z_angle_min) / (lidar_param.z_angle_max - lidar_param.z_angle_min) * (lidar_param.range_image_height - 1)).astype(np.int32)
        depth = np.linalg.norm(pointcloud[:, :3], 2, -1)
        depth[np.where(depth < lidar_param.nearest_range)] = lidar_param.nearest_range
        depth[np.where(depth > lidar_param.furthest_range)] = lidar_param.furthest_range
        range_image[row, col] = depth
        return range_image

    @staticmethod
    def range_image_to_point_cloud_batch(range_image, transform_map):
        bs, h, w, _ = range_image.shape
        point_cloud = torch.zeros((bs, h, w, 3), device=range_image.device)
        for b in range(bs):
            point_cloud[b] = PCTransformer.range_image_to_point_cloud(range_image[b], transform_map)
        return point_cloud

    @staticmethod
    def range_image_to_point_cloud(range_image, transform_map):
        # max_range = lidar_param.furthest_range
        # min_range = lidar_param.nearest_range
        # depth = range_image * (max_range - min_range) + min_range
        # transform_map = lidar_param.range_image_to_point_cloud_map
        point_cloud = range_image * transform_map
        return point_cloud


if __name__ == '__main__':
    from utils import utils
    from utils.visualize_utils import save_point_cloud_to_pcd

    train_datalist = 'data/train_64E_KITTI.txt'
    lidar_yaml = 'cfgs/vlp64_KITTI.yaml'

    # train_datalist = 'data/train_64E.txt'
    # lidar_yaml = 'cfgs/vlp64.yaml'

    # train_datalist = 'data/train.txt'
    # lidar_yaml = 'cfgs/vlp16.yaml'

    # train_datalist = 'data/train_64E_waymo.txt'
    # lidar_yaml = 'cfgs/vlp64_waymo.yaml'

    lidar = LidarParam(lidar_yaml)
    dataset = Dataset(datalist=train_datalist, lidar_param=lidar)
    # IPython.embed()
    pc, ri = dataset[0]
    # pc = PCTransformer.range_image_to_point_cloud(ri, lidar)
    print('read range image: ', dataset.data_list[0])
    IPython.embed()
    #
    # path = 'data/pc_test.pcd'
    # pc_save = pc.reshape(-1, 3)
    # save_point_cloud_to_pcd(pc_save, path)
    # ri_save = ri[:, :, 0]
    # path = 'data/ri_test.png'
    # utils.save_range_image_to_png(ri_save, path)

    def test_time(dataset, n=None):
        if n is None:
            n = len(dataset)
        start = time.time()
        for i in range(n):
            res = dataset[i]
        time_cost = time.time() - start
        print('time cost: ', time_cost)
        return time_cost

    dataset = Dataset(datalist=train_datalist, lidar_param=lidar, use_radius_outlier_removal=True)
    time_cost = test_time(dataset, n=10)
    print('when use radius outlier removal: ', time_cost)
    dataset = Dataset(datalist=train_datalist, lidar_param=lidar, use_radius_outlier_removal=False)
    time_cost = test_time(dataset, n=10)
    print('when do not use radius outlier removal: ', time_cost)
    # range_image_double = np.zeros((64, 2000, 1))
    # point_world = [10, 10, -2]
    # [x_world, y_world, z_world] = point_world
    # from utils.pointcloud_to_rangeimage.pointcloud_to_rangeimage import *
    # degree_in_xy = calc_theta_degree_in_xy(point_world)
    # y_rangeimage = math.floor(degree_in_xy / 360 * 2000)
    # degree_in_z = calc_theta_degree_in_z(point_world)
    # x_rangeimage = math.floor((2.5 - degree_in_z) / (2.5 - -24.9) * 64)
    # x_rangeimage = int((degree_in_z - -24.9) / ((2.5 - -24.9) / 64))
    # y_rangeimage = int(degree_in_xy / (360/2000)) % 2000
    #
    # d = math.sqrt(x_world * x_world + y_world * y_world + z_world * z_world)
    # range_image_double[x_rangeimage][y_rangeimage][0] = d
    #
    # pc_new = PCTransformer.range_image_to_point_cloud(range_image_double)
    IPython.embed()