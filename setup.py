##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import io
import os
import glob
import subprocess

from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension

cwd = os.path.dirname(os.path.abspath(__file__))

version = '1.2.2'
try:
    if not os.getenv('RELEASE'):
        from datetime import date
        today = date.today()
        day = today.strftime("b%Y%m%d")
        version += day
except Exception:
    pass

def create_version_file():
    global version, cwd
    print('-- Building version ' + version)
    version_path = os.path.join(cwd, 'encoding', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is encoding version file."""\n')
        f.write("__version__ = '{}'\n".format(version))

requirements = [
    'numpy',
    'tqdm',
    'nose',
    'portalocker',
    'torch>=1.4.0',
    'torchvision>=0.5.0',
    'Pillow',
    'scipy',
    'requests',
    'portalocker',
]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))

    cpu_extensions_dir = os.path.join(this_dir, "encoding", "lib", "cpu")
    gpu_extensions_dir = os.path.join(this_dir, "encoding", "lib", "gpu")

    source_cpu = glob.glob(os.path.join(cpu_extensions_dir, "*.cpp"))
    source_cuda = glob.glob(os.path.join(gpu_extensions_dir, "*.cpp")) + \
        glob.glob(os.path.join(gpu_extensions_dir, "*.cu"))

    print('c++: ', source_cpu)
    print('cuda: ', source_cuda)

    sources = source_cpu

    extra_compile_args = {"cxx": []}
    include_dirs = [cpu_extensions_dir]

    ext_modules = [
        CppExtension(
            "encoding.cpu",
            source_cpu,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
        )
    ]

    if CUDA_HOME is not None:
        define_macros = [("WITH_CUDA", None)]
        include_dirs += [gpu_extensions_dir]
        extra_compile_args["nvcc"] = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
        ]

        ext_modules.extend([
            CUDAExtension(
                "encoding.gpu",
                source_cuda,
                include_dirs=include_dirs,
                define_macros=define_macros,
                extra_compile_args=extra_compile_args,
            )
        ])

    return ext_modules

if __name__ == '__main__':
    create_version_file()
    setup(
        name="torch-encoding",
        version=version,
        author="Hang Zhang",
        author_email="zhanghang0704@gmail.com",
        url="https://github.com/zhanghang1989/PyTorch-Encoding",
        description="PyTorch Encoding Package",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license='MIT',
        install_requires=requirements,
        packages=find_packages(exclude=["tests", "experiments"]),
        package_data={ 'encoding': [
            'LICENSE',
            'lib/cpu/*.h',
            'lib/cpu/*.cpp',
            'lib/gpu/*.h',
            'lib/gpu/*.cpp',
            'lib/gpu/*.cu',
        ]},
        ext_modules=get_extensions(),
        cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
    )
