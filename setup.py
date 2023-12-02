#! -*- coding: utf-8 -*-
from setuptools import setup, find_packages

ignore = ['test','tests']

install_requires = [
    'lightning>=2.0 , <50.0',
    'numpy-io>=0.0.10 , < 0.1.0',
    'sentencepiece',
    'numpy',
    'transformers>=4.22',
    'seqmetric',
    'scipy',
    'scikit-learn',
    'tensorboard',
    'tqdm',
    'six',
    'safetensors',
    'fastdatasets>=0.9.17',
]
setup(
    name='deep_training',
    version='0.2.10',
    description='an easy training architecture',
    long_description='torch_training: https://github.com/ssbuild/deep_training.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/deep_training',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=install_requires,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["**/*.cu", "**/*.cpp", "**/*.cuh", "**/*.h", "**/*.pyx"]},

)
