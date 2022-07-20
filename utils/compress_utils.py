import IPython
import os
import yaml
import numpy as np
import copy
from easydict import EasyDict
from pathlib import Path
# import pyzfp  # TODO
import bz2
import gzip
import lz4  ## lz4 version is 0.7.0 or install lz4 from source
# numpy to bytes
# https://stackoverflow.com/questions/62352670/deserialization-of-large-numpy-arrays-using-pickle-is-order-of-magnitude-slower?noredirect=1#comment110277408_62352670
# https://stackoverflow.com/questions/53376786/convert-byte-array-back-to-numpy-array
from utils.contour_utils import ContourExtractor


def load_compressor_cfg(yaml_file):
    with open(yaml_file, 'r') as f:
        try:
            config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            config = yaml.load(f)
    compressor_config = EasyDict(config)
    return compressor_config


def calc_residual(range_image, plane_param, cluster_param, cluster_idx, lidar_param):
    cluster_num = cluster_param.shape[0]
    plane_num = plane_param.shape[0]
    transform_map = lidar_param.range_image_to_point_cloud_map

    residual = np.zeros((range_image.shape[0], range_image.shape[1]), np.float32)
    for c in range(cluster_num):
        chosen_idx = np.where(cluster_idx == c)
        if chosen_idx[0].shape[0] > 0:
            residual[chosen_idx] = range_image[chosen_idx][..., 0] - np.linalg.norm(cluster_param[c], 2, -1)

    for p in range(plane_num):
        chosen_idx = np.where(cluster_idx == cluster_num + p)
        if chosen_idx[0].shape[0] > 0:
            r_plane = plane_param[p, 3] / np.sum(plane_param[p, :3] * transform_map[chosen_idx], -1)
            residual[chosen_idx] = range_image[chosen_idx][..., 0] - r_plane
    return residual


def compress_point_cloud(point_cloud, range_image, plane_param, cluster_xyz, cluster_feats, cluster_idx, residual, accuracy, basic_compressor, full=False):
    # cluster_num = cluster_xyz.shape[0]
    # plane_num = plane_param.shape[0]
    #


    # zero_idx = np.where(range_image[..., 0] == 0)
    # cluster_idx[zero_idx] = cluster_num + plane_num
    # residual[zero_idx] = 0
    # residual_unit[zero_idx] = 0

    original_data = {}
    if full:
        original_data['point_cloud'] = point_cloud
        original_data['range_image'] = range_image

    if residual is not None:
        residual_unit = np.rint(residual / accuracy)
        original_data['residual_unit'] = residual_unit.astype(np.int16)

        if full:
            idx = np.where(cluster_idx == 0)
            original_data['plane_residual'] = original_data['residual_unit'][idx]
            idx = np.where(cluster_idx > 0)
            original_data['cluster_residual'] = original_data['residual_unit'][idx]

    contour_map, idx_sequence = ContourExtractor.extract_contour(cluster_idx)
    contour_map = contour_map.astype(np.bool)
    contour_map = np.packbits(contour_map, axis=None)  # same as before
    original_data['contour_map'] = contour_map.astype(np.uint8)
    original_data['idx_sequence'] = idx_sequence.astype(np.uint16)
    if plane_param is not None:
        original_data['plane_param'] = plane_param.astype(np.float32)
    original_data['cluster_xyz'] = cluster_xyz.astype(np.float32)
    if cluster_feats is not None:
        original_data['cluster_feats'] = cluster_feats.astype(np.float32)

    compressed_data = basic_compressor.compress_dict(original_data)
    return original_data, compressed_data


def decompress_point_cloud(compressed_data, basic_compressor, cluster_num, plane_num, accuracy, lidar_param):
    decompressed_data = basic_compressor.decompress_dict(compressed_data)
    residual_unit = np.ndarray(shape=(lidar_param.range_image_height, lidar_param.range_image_width), dtype=np.int16,
                          buffer=decompressed_data['residual_unit'])
    residual = residual_unit * accuracy
    plane_param = np.ndarray(shape=(plane_num, 4), dtype=np.float32, buffer=decompressed_data['plane_param'])
    cluster_xyz = np.ndarray(shape=(cluster_num, 3), dtype=np.float32, buffer=decompressed_data['cluster_xyz'])
    if 'cluster_feats' in decompressed_data.keys():
        cluster_feats = np.ndarray(shape=(cluster_num, 3), dtype=np.float32, buffer=decompressed_data['cluster_feats'])
    else:
        cluster_feats = None
    contour_map = np.ndarray(shape=(-1,), dtype=np.uint8, buffer=decompressed_data['contour_map'])
    contour_map = np.unpackbits(contour_map)
    contour_map = np.reshape(contour_map, (lidar_param.range_image_height, lidar_param.range_image_width))
    idx_sequence = np.ndarray(shape=(-1,), dtype=np.uint16, buffer=decompressed_data['idx_sequence'])
    idx_map = ContourExtractor.recover_map(contour_map, idx_sequence)
    return residual, idx_map, plane_param, cluster_xyz, cluster_feats


def compress_plane_idx_map(plane_idx, single_line=True):
    if not single_line:
        from utils.contour_utils import ContourExtractorDoubleDirection
        contour_map, idx_sequence = ContourExtractorDoubleDirection.extract_contour(plane_idx)
        # sorted_idx_map, sorted_compressed_idx, original_compressed_idx = FF.sorted_index_encoder()

        contour_map = contour_map.astype(np.bool)  # shape=(range_image.shape[0], range_image.shape[1], 2), dtype=np.bool
        # for boolean data, packbits can save 1/8 (8 boolean to 1 uint8)
        contour_map = np.packbits(contour_map, axis=None)  # shape=(-1,), dtype=np.uint8
        # back = np.ndarray(shape=(-1,), dtype=np.uint8, buffer=back_compressed)
        # back = np.unpackbits(back)
        # back = np.reshape(back, (64, 2000, 2))
    else:
        from utils.contour_utils import ContourExtractor
        contour_map, idx_sequence = ContourExtractor.extract_contour(plane_idx)
        contour_map = contour_map.astype(np.bool)
        contour_map = np.packbits(contour_map, axis=None)  # same as before
    return contour_map, idx_sequence


class BasicCompressor:
    def __init__(self, compressor_yaml=None, method_name=None):
        if compressor_yaml is not None:
            with open(compressor_yaml, 'r') as f:
                try:
                    config = yaml.load(f, Loader=yaml.FullLoader)
                except:
                    config = yaml.load(f)
            compressor_config = EasyDict(config)
            self.method_name = compressor_config.BASIC_COMPRESSOR_NAME
        if method_name is not None:
            self.method_name = method_name

        assert self.method_name in ['lz4', 'bzip2', 'gzip', 'deflate'], \
            'Compression method is not existed. (lz4, bzip2, gzip, deflate)'

    def compress_dict(self, data_dict):
        compressed_data_dict = copy.deepcopy(data_dict)
        for key, val in data_dict.items():
            compressed_data_dict[key] = self.compress(val)
        return compressed_data_dict

    def decompress_dict(self, data_dict):
        decompressed_data_dict = copy.deepcopy(data_dict)
        for key, val in data_dict.items():
            decompressed_data_dict[key] = self.decompress(val)
        return decompressed_data_dict

    def compress(self, np_array):
        if self.method_name == 'lz4':
            return self.lz4_compress(np_array)
        if self.method_name == 'bzip2':
            return self.bzip2_compress(np_array)
        if self.method_name == 'gzip' or 'deflate':
            return self.gzip_compress(np_array)

    def decompress(self, np_array):
        if self.method_name == 'lz4':
            return self.lz4_decompress(np_array)
        if self.method_name == 'bzip2':
            return self.bzip2_decompress(np_array)
        if self.method_name == 'gzip' or 'deflate':
            return self.gzip_decompress(np_array)

    @staticmethod
    def lz4_compress(np_array):
        return lz4.dumps(np_array)

    @staticmethod
    def lz4_decompress(data):
        return lz4.loads(data)

    @staticmethod
    def bzip2_compress(np_array):
        return bz2.compress(np_array)

    @staticmethod
    def bzip2_decompress(data):
        return bz2.decompress(data)

    @staticmethod
    def gzip_compress(np_array):
        return gzip.compress(np_array)

    @staticmethod
    def gzip_decompress(data):
        return gzip.decompress(data)


if __name__ == "__main__":
    import numpy as np
    from time import time

    data_type = np.int8
    data_size = (100, 1000)
    rand_array = np.random.randint(50, size=data_size)
    rand_bytes = rand_array.astype(data_type).tobytes()

    methods = ['lz4', 'bzip2', 'gzip']
    for method in methods:
        print('\nTest ', method)
        BC = BasicCompressor(method_name=method)
        repeat_time = 10
        t0 = time()
        for i in range(repeat_time):
            compressed_data = BC.compress(rand_bytes)
        t1 = time()
        for i in range(repeat_time):
            decompressed_data = BC.decompress(compressed_data)
        print('%d times compress cost time: %.04f, decompress cost time: %.04f' % (repeat_time, t1 - t0, time() - t1))
        print('Compression rate: ', len(rand_bytes) / len(compressed_data))
        recovered = np.ndarray(shape=data_size, dtype=data_type, buffer=decompressed_data)
        assert np.array_equal(recovered, rand_array), '%s is not working.' % method
    print('All compression methods are working.')
