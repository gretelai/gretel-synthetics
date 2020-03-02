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
        'tensorflow==2.1.0',
        'tensorflow_privacy==0.2.2',
        'smart_open'
    ]
)
