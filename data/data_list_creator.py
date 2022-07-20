import os
import IPython
import glob


output_name = 'test_64E_KITTI_campus.txt'

kitti_raw_data_root = '/data/KITTI_rawdata/test/campus'

dirs = os.listdir(kitti_raw_data_root)
dirs.sort()
flag = False
txt_file = open(output_name, 'w')
for d in dirs:
    files = glob.glob(os.path.join(kitti_raw_data_root, d, '*/*/velodyne_points/data/*.bin'))
    files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    for f in files:
        str_write = f + '\n'
        txt_file.write(str_write)
        flag = True
assert flag
txt_file.close()
