import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import IPython
import copy

def delete_all_files(dir):
    filelist = os.listdir(dir)  # 列出该目录下的所有文件名
    for f in filelist:
        filepath = os.path.join(dir, f)  # 将文件名映射成绝对路劲
        if os.path.isfile(filepath):  # 判断该文件是否为文件或者文件夹
            os.remove(filepath)  # 若为文件，则直接删除
            print(str(filepath) + " removed!")
    print("remove all old test log files from ", dir)



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