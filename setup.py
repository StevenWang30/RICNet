import os
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
try:
   from setupext_janitor import janitor
   CleanCommand = janitor.CleanCommand
except ImportError:
   CleanCommand = None

cmd_classes = {'build_ext': BuildExtension}
if CleanCommand is not None:
   cmd_classes['clean'] = CleanCommand


def make_cuda_ext(name, module, sources):
    cuda_ext = CUDAExtension(
        name='%s.%s' % (module, name),
        sources=[os.path.join(*module.split('.'), src) for src in sources]
    )
    return cuda_ext


if __name__ == '__main__':
    setup(
        name='ricnet',
        description='Three-stage range image-based end-to-end point cloud compression.',
        install_requires=[
            'numpy',
            'torch>=1.7',
            'tensorboardX',
            'easydict',
            'pyyaml'
        ],
        author='Sukai Wang',
        author_email='swangcy@connect.com',
        license='Apache License 2.0',
        packages=find_packages(),
        cmdclass=cmd_classes,
        ext_modules=[
            make_cuda_ext(
                name='pointnet2_batch_cuda',
                module='utils.ops.pointnet2.pointnet2_batch',
                sources=[
                    'src/pointnet2_api.cpp',
                    'src/ball_query.cpp',
                    'src/ball_query_gpu.cu',
                    'src/group_points.cpp',
                    'src/group_points_gpu.cu',
                    'src/interpolate.cpp',
                    'src/interpolate_gpu.cu',
                    'src/sampling.cpp',
                    'src/sampling_gpu.cu',
                ],
            ),
        ],
        setup_requires=['setupext_janitor'],
        entry_points={
            # normal parameters, ie. console_scripts[]
            'distutils.commands': [
                ' clean = setupext_janitor.janitor:CleanCommand']
        }
    )
