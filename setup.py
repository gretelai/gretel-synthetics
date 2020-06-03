from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))

with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(path.join(this_dir, 'VERSION')) as f:
    version = f.read()


setup(
    name='gretel-synthetics',
    version=version,
    description='Synthetic Data Generation with optional Differential Privacy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    install_requires=[
        'tensorflow_privacy==0.2.2',
        'sentencepiece==0.1.91',
        'smart_open==1.10.0',
        'tqdm<5.0',
        'pandas==1.0.3',
        'numpy==1.18.3'
    ],
    extras_require={
        'tf': ['tensorflow==2.1.0'],
        'py36': ['dataclasses==0.7']
    }
)
