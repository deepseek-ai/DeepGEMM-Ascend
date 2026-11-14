import ast
import os
import re
import shutil
import setuptools
import subprocess
import torch
from pathlib import Path
from setuptools import find_packages
from setuptools.command.build_py import build_py
from torch.utils.cpp_extension import CppExtension, BuildExtension
from scripts.generate_pyi import generate_pyi_file


DG_SKIP_BUILD = int(os.getenv('DG_SKIP_BUILD', '0')) == 1
DG_USE_LOCAL_VERSION = int(os.getenv('DG_USE_LOCAL_VERSION', '1')) == 1
persistent_env_names = (
    'DG_JIT_CACHE_DIR',
    'DG_JIT_DEBUG',
    'DG_JIT_DUMP_ASM',
    'DG_JIT_KERNEL_DEBUG_INFO',
    'DG_JIT_LAUNCH_TIMEOUT',
    'DG_JIT_PRINT_COMPILER_COMMAND',
    'DG_JIT_PRINT_LOAD_TIME',
)

# Ascend toolkit discovery
ASCEND_HOME = os.environ.get('ASCEND_HOME_PATH',
    os.environ.get('ASCEND_TOOLKIT_HOME', '/usr/local/Ascend/ascend-toolkit/latest'))

# torch_npu paths
import torch_npu
TORCH_NPU_DIR = os.path.dirname(torch_npu.__file__)

# Compiler flags
cxx_flags = [
    '-std=c++20', '-O3', '-g1', '-fPIC', '-Wno-psabi', '-Wno-deprecated-declarations',
    f'-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}'
]

# Sources
current_dir = os.path.dirname(os.path.realpath(__file__))
sources = [ 'csrc/python_api.cpp' ]
build_include_dirs = [
    f'{ASCEND_HOME}/include',
    f'{TORCH_NPU_DIR}/include',
    f'{TORCH_NPU_DIR}/include/third_party/acl/inc',

    os.path.join(current_dir, 'deep_gemm/include'),
    os.path.join(current_dir, 'third-party/deep-jit/include'),
    os.path.join(current_dir, 'third-party/magic_enum/include'),
]
build_libraries = ['ascendcl', 'hccl_fwk', 'torch_npu', 'opapi', 'dl']
build_library_dirs = [
    f'{ASCEND_HOME}/lib64',
    f'{TORCH_NPU_DIR}/lib',
]


def get_package_version():
    with open(Path(current_dir) / 'deep_gemm' / '__init__.py', 'r') as f:
        version_match = re.search(r'^__version__\s*=\s*(.*)$', f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))

    revision = ''
    if DG_USE_LOCAL_VERSION:
        # noinspection PyBroadException
        try:
            status_cmd = ['git', 'status', '--porcelain']
            status_output = subprocess.check_output(status_cmd).decode('ascii').strip()
            if status_output:
                print(f'Warning: Git working directory is not clean. Uncommitted changes:\n{status_output}')
                assert False, 'Git working directory is not clean'

            cmd = ['git', 'rev-parse', '--short', 'HEAD']
            revision = '+' + subprocess.check_output(cmd).decode('ascii').rstrip()
        except Exception:
            revision = '+local'
    return f'{public_version}{revision}'


def get_ext_modules():
    if DG_SKIP_BUILD:
        return []

    return [CppExtension(name='deep_gemm._C',
                         sources=sources,
                         include_dirs=build_include_dirs,
                         libraries=build_libraries,
                         library_dirs=build_library_dirs,
                         extra_compile_args=cxx_flags,
                         extra_link_args=['-Wl,-Bsymbolic-functions'])]


class CustomBuildPy(build_py):
    def run(self):
        # Generate envs.py
        self.generate_default_envs()

        # Generate .pyi stub file
        self.generate_pyi_file()

        # Copy csrc and docs into the wheel for agent-side error lookup
        self.prepare_agent_files()

        # Run the regular build
        build_py.run(self)

    def prepare_agent_files(self):
        # Copy csrc and docs into the wheel for agent-side error lookup
        package_dir = os.path.join(self.build_lib, 'deep_gemm')
        for name in (
            'csrc',
        ):
            dst = os.path.join(package_dir, name)
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(os.path.join(current_dir, name), dst)

    def generate_pyi_file(self):
        generate_pyi_file(name='_C', root='./csrc/apis', output_dir='./stubs')
        pyi_source = os.path.join(current_dir, 'stubs', '_C.pyi')
        pyi_target = os.path.join(self.build_lib, 'deep_gemm', '_C.pyi')

        if os.path.exists(pyi_source):
            print(f"Copying .pyi file from {pyi_source} to {pyi_target}")
            os.makedirs(os.path.dirname(pyi_target), exist_ok=True)
            shutil.copy2(pyi_source, pyi_target)
        else:
            print(f"Warning: .pyi file not found at {pyi_source}")

    def generate_default_envs(self):
        code = '# Pre-installed environment variables\n'
        code += 'persistent_envs = dict()\n'
        for name in persistent_env_names:
            code += f"persistent_envs['{name}'] = '{os.environ[name]}'\n" if name in os.environ else ''

        envs_dir = os.path.join(self.build_lib, 'deep_gemm')
        os.makedirs(envs_dir, exist_ok=True)
        with open(os.path.join(envs_dir, 'envs.py'), 'w') as f:
            f.write(code)


if __name__ == '__main__':
    # noinspection PyTypeChecker
    setuptools.setup(
        name='deep_gemm',
        version=get_package_version(),
        packages=find_packages('.'),
        extras_require={'tilelang': ['tilelang']},
        package_data={
            'deep_gemm': [
                'include/**/*',
            ]
        },
        ext_modules=get_ext_modules(),
        zip_safe=False,
        cmdclass={
            'build_py': CustomBuildPy,
            'build_ext': BuildExtension,
        },
    )
