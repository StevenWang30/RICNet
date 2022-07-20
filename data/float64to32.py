import glob
import IPython
import os
import numpy as np

root = './navie_refinement'
dirs = glob.glob(os.path.join(root, '*'))
dirs.sort()

for dir in dirs:
    files = glob.glob(os.path.join(dir, '*'))
    for file in files:
        point_cloud = np.fromfile(file, dtype=np.float64)
        point_cloud = point_cloud.reshape((-1, 4))
        point_cloud.astype(np.float32).tofile(file)
        print(file)
            
IPython.embed()
