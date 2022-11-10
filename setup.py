#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='torch_training',
    version='0.11.3',
    description='an easy training architecture',
    long_description='torch_training: https://github.com/ssbuild/torch_training.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/torch_training',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=['pytorch-lightning>=1.7','fastdataset>=0.7.6','numpy','transformer'],
    packages=find_packages()
)
