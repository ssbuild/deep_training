#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='deep_training',
    version='0.0.1',
    description='an easy training architecture',
    long_description='torch_training: https://github.com/ssbuild/deep_training.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/deep_training',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=['pytorch-lightning>=1.7','fastdataset>=0.8.1','numpy',
                      'transformer','seqmetric','sklearn','scipy','scikit-learn'],
    packages=find_packages()
)
