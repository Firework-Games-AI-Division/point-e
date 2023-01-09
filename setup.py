from setuptools import setup
import importlib
import glob
import os
import subprocess
import logging
import pip
cwd = os.path.dirname(os.path.abspath(__file__))

# pip.main(['install', 'torch', 'torchvision', '--extra-index-url', 'https://download.pytorch.org/whl/cu117'])
import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME

logger = logging.getLogger()
logging.basicConfig(format='%(levelname)s - %(message)s')



def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([os.path.join(cuda_dir, "bin", "nvcc"), "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    release = output[release_idx].split(".")
    bare_metal_major = release[0]
    bare_metal_minor = release[1][0]
    return raw_output, bare_metal_major, bare_metal_minor


def get_include_dirs():
    include_dirs = []
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        _, bare_metal_major, _ = get_cuda_bare_metal_version(CUDA_HOME)
        if "CUB_HOME" in os.environ:
            logging.warning(f'Including CUB_HOME ({os.environ["CUB_HOME"]}).')
            include_dirs.append(os.environ["CUB_HOME"])
        else:
            if int(bare_metal_major) < 11:
                logging.warning(f'Including default CUB_HOME ({os.path.join(cwd, "third_party", "cub")}).')
                include_dirs.append(os.path.join(cwd, 'third_party', 'cub'))

    return include_dirs

def get_extensions():

    extra_compile_args = {'cxx': ['-O3']}
    define_macros = []
    include_dirs = []
    sources = glob.glob('point_e/util/dmtet/csrc/**/*.cpp', recursive=True)

    # FORCE_CUDA is for cross-compilation in docker build
    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        with_cuda = True
        define_macros += [("WITH_CUDA", None), ("THRUST_IGNORE_CUB_VERSION_CHECK", None)]
        sources += glob.glob('point_e/util/dmtet/csrc/**/*.cu', recursive=True)
        extension = CUDAExtension
        extra_compile_args.update({'nvcc': [
            '-O3',
            '-DWITH_CUDA',
            '-DTHRUST_IGNORE_CUB_VERSION_CHECK'
        ]})
        include_dirs = get_include_dirs()
    else:
        extension = CppExtension
        with_cuda = False
    extensions = []
    extensions.append(
        extension(
            name='point_e._C',
            sources=sources,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            include_dirs=include_dirs
        )
    )

    # use cudart_static instead
    for extension in extensions:
        extension.libraries = ['cudart_static' if x == 'cudart' else x
                               for x in extension.libraries]

    return extensions
  
setup(
    name="point-e",
    packages=[
        "point_e",
        "point_e.diffusion",
        "point_e.evals",
        "point_e.models",
        "point_e.util",
        "point_e.commands",
        "point_e.util.dmtet"
    ],
    install_requires=[
        "filelock",
        "fire",
        "humanize",
        "requests",
        "tqdm",
        "matplotlib",
        "scikit-image",
        "scipy",
        "clip @ git+https://github.com/Firework-Games-AI-Division/CLIP.git",
    ],
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext --inplace': BuildExtension.with_options(no_python_abi_suffix=True)
    },
    author="OpenAI",
)
