from setuptools import setup, find_packages
import unittest

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

ext_modules=[
    CUDAExtension('mesh_grid', [
        'mesh_grid.cpp',
        'mesh_grid_kernel.cu',
        ]),
    ]

setup(
    name='mesh_grid',
    version='1.0.0',
    description='Fast Conversion from Triangular Mesh to SDF',
    author='Wei Cheng, Su Xu, Jingtan Piao, Chen Qian, Wayne Wu, Kwan-Yee Lin, Hongsheng Li',
    author_email='wchengad@connect.ust.hk',
    url='https://github.com/generalizable-neural-performer/gnr/',
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension},
    scripts=['mesh_grid_searcher.py']
)
