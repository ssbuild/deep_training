#! -*- coding: utf-8 -*-

from setuptools import setup, find_packages

ignore = ['test','tests']
setup(
    name='deep_training',
    version='0.0.12',
    description='an easy training architecture',
    long_description='torch_training: https://github.com/ssbuild/deep_training.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/deep_training',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=['pytorch-lightning>=1.7,<1.10',
                      'fastdatasets>=0.9.4 , <= 1',
                      'tfrecords >= 0.2.4 , <=3',
                      'numpy',
                      'transformers','seqmetric','sklearn','scipy','scikit-learn','tqdm','six','jieba'],
    packages=[p for p in find_packages() if p not in ignore]
)
