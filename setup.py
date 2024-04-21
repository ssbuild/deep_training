#! -*- coding: utf-8 -*-
from setuptools import setup, find_packages

ignore = ['test','tests']

install_requires = [
    'lightning>=2.0 , <50.0',
    'numpy-io>=0.0.10 , < 0.1.0',
    'sentencepiece',
    'numpy',
    'transformers>=4.39',
    'seqmetric',
    'scipy',
    'scikit-learn',
    'tensorboard',
    'tqdm',
    'six',
    'pyyaml'
    'safetensors',
    'fastdatasets>=0.9.17',
]
setup(
    name='deep_training',
    version='0.3.0.rc0',
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

    entry_points={
        'console_scripts': [
            'deep_export = deep_training.tools.export_transformers:export',
        ],
    }

)
