from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import sys
import platform

# Make sure we're using the correct architecture
is_64bit = sys.maxsize > 2**32
if not is_64bit:
    raise RuntimeError('Only 64-bit architecture is supported')

# Force Visual Studio to use x64 host tools
if sys.platform == 'win32':
    os.environ['PreferredToolArchitecture'] = 'x64'

# Get current directory
this_dir = os.path.dirname(os.path.abspath(__file__))

# CUDA compilation flags
nvcc_flags = [
    '-O3',
    '--use_fast_math',
    '-gencode=arch=compute_89,code=sm_89',  # RTX 4090
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
]

# Add platform-specific flags
if sys.platform == 'win32':
    nvcc_flags += [
        '--compiler-options=/O2',
        '--compiler-options=/MD',
        '--compiler-options=/EHsc',
        '--compiler-options=/std:c++17',
    ]
else:
    nvcc_flags += [
        '-std=c++17',
    ]

# Define the extension module
ext_modules = [
    CUDAExtension(
        name='quaternion_ops',  
        sources=[
            os.path.join(this_dir, 'quaternion_ops_py.cpp'),
            os.path.join(this_dir, 'quaternion_ops.cu'),
        ],
        extra_compile_args={
            'cxx': ['/std:c++17'] if sys.platform == 'win32' else ['-std=c++17'],
            'nvcc': nvcc_flags,
        },
        define_macros=[
            ('TORCH_EXTENSION_NAME', 'quaternion_ops'),
            ('_GLIBCXX_USE_CXX11_ABI', '0'),
        ],
    )
]

setup(
    name='quaternion_ops',
    version='0.1',
    author='Bryce Grant',
    author_email='bag100@case.edu',
    description='CUDA extensions for quaternion operations',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=False)
    },
)
