import os
import torch
from torch.utils.cpp_extension import load

cwd = os.path.dirname(os.path.realpath(__file__))
cpu_path = os.path.join(cwd, 'cpu')
gpu_path = os.path.join(cwd, 'gpu')

cpu = load( 'enclib_cpu', [
        os.path.join(cpu_path, 'roi_align.cpp'),
        os.path.join(cpu_path, 'roi_align_cpu.cpp'),
    ], build_directory=cpu_path, verbose=False)

if torch.cuda.is_available():
    gpu = load( 'enclib_gpu', [
            os.path.join(gpu_path, 'operator.cpp'),
            os.path.join(gpu_path, 'encoding_kernel.cu'),
            os.path.join(gpu_path, 'syncbn_kernel.cu'),
            os.path.join(gpu_path, 'roi_align_kernel.cu'),
        ], build_directory=gpu_path, verbose=False)
