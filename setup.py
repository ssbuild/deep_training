#! -*- coding: utf-8 -*-
import re

from setuptools import setup, find_packages

packge_list = find_packages('src')
package_dir= {'deep_training.' + k : 'src/' + k.replace('.','/') for k in packge_list }
package_dir.update({'deep_training': 'src'})

ignore = ['test','tests']
setup(
    name='deep_training',
    version='0.1.12.post2',
    description='an easy training architecture',
    long_description='torch_training: https://github.com/ssbuild/deep_training.git',
    license='Apache License 2.0',
    url='https://github.com/ssbuild/deep_training',
    author='ssbuild',
    author_email='9727464@qq.com',
    install_requires=['lightning>=2.0',
                      'numpy-io>=0.0.7 , < 0.1.0',
                      'sentencepiece',
                      'numpy',
                      'transformers>=4.22',
                      'seqmetric',
                      'scipy',
                      'scikit-learn',
                      'tensorboard',
                      'tqdm',
                      'six'],

    packages=list(package_dir.keys()),
    package_dir= package_dir,
    package_data={'': ['nlp/models/rwkv4/cuda/*.cu', 'nlp/models/rwkv4/cuda/*.c', 'nlp/models/rwkv4/cuda/*.cpp'],},

)
