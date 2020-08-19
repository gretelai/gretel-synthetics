from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))

with open(path.join(this_dir, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# with open(path.join(this_dir, 'VERSION')) as f:
#    version = f.read()


setup(
    name='gretel-synthetics',
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description='Synthetic Data Generation with optional Differential Privacy',
    long_description=long_description,
    long_description_content_type='text/markdown',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    python_requires=">=3.6",
    install_requires=[
        'tensorflow_privacy==0.2.2',
        'sentencepiece==0.1.91',
        'smart_open==2.0.0',
        'tqdm<5.0',
        'pandas>=1.0.0',
        'numpy>=1.18.0',
        'dataclasses==0.7;python_version<"3.7"',
        'cloudpickle==1.5.0',
    ],
    extras_require={
        'tf': ['tensorflow==2.1.0']
    }
)
