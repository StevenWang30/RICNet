# Point Cloud Compression with Range Image-based Entropy Model for Autonomous Driving

A range image-based three-stage framework to compress the scanning LiDAR’s point clouds using the entropy model.

## Overview
- [Installation](#installation)
- [Getting Started](#get-started)

## Installation (using docker)

1. docker build
```
cd .../RICNet/docker/dockerfile
docker build -t ricnet_docker .
docker run --gpus all -it -p 6001:6006 -v /tmp/.X11-unix:/tmp/.X11-unix -v /home/skwang/DLProject/Pytorch/code:/code -v /data:/data -e DISPLAY=:1 --name ricnet_proj ricnet_docker:latest /bin/bash
```

2. install package
```
python setup.py develop 
```

3. install MinkowskiEngine
```
pip install git+https://github.com/NVIDIA/MinkowskiEngine.git --install-option="--blas=openblas" --install-option="--force_cuda"
```

4. Uninstall: python3 setup.py develop --uninstall --user (In host)

## Get Started

### Training
```
python train.py
```

### Compress and Decompress
```
python compress.py
```

and 

```
python decompress.py
```

Noted that the model setting must be the same as the training model. 

## Acknowledgement
Thanks for the code openpcdet.