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
    ext_modules=ext_modules,
    cmdclass = {'build_ext': BuildExtension}
)
