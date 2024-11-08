from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import subprocess
import os

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        # Assuming the setup.py is located at the project root which also contains the CMakeLists.txt
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        # Determine the number of cores to use
        num_cores = os.cpu_count()  # Get the number of cores available on your system

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp)
        subprocess.check_call(['cmake', '--build', '.', '--config', cfg, '--', f'-j{num_cores}'], cwd=self.build_temp)  # Adjusted line

setup(
    name='pysuperansac',
    version='0.1',
    author='Daniel Barath',
    description='A RANSAC implementation for robust estimation.',
    long_description='',
    ext_modules=[CMakeExtension('SupeRANSAC', sourcedir='.')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
)
