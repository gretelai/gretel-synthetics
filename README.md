# Gretel Synthetics

<p align="center">
    <a href="https://gretel.ai"><img width="128px" src="https://gretel-public-website.s3.amazonaws.com/assets/gobs_the_cat_@1x.png" alt="Gobs the Gretel.ai cat" /></a><br />
    <i>A permissive synthetic data library from Gretel.ai</i>
</p>

[![Documentation Status](https://readthedocs.org/projects/gretel-synthetics/badge/?version=stable)](https://gretel-synthetics.readthedocs.io/en/stable/?badge=stable)
[![CLA assistant](https://cla-assistant.io/readme/badge/gretelai/gretel-synthetics)](https://cla-assistant.io/gretelai/gretel-synthetics)
[![PyPI](https://badge.fury.io/py/gretel-synthetics.svg)](https://badge.fury.io/py/gretel-synthetics)
[![Python](https://img.shields.io/pypi/pyversions/gretel-synthetics.svg)](https://github.com/gretelai/gretel-synthetics)
[![Downloads](https://pepy.tech/badge/gretel-synthetics)](https://pepy.tech/project/gretel-synthetics)
[![GitHub stars](https://img.shields.io/github/stars/gretelai/gretel-synthetics?style=social)](https://github.com/gretelai/gretel-synthetics)
[![Discord](https://img.shields.io/discord/1007817822614847500?label=Discord&logo=Discord)](https://gretel.ai/discord)

## Documentation

- [Get started with gretel-synthetics](https://gretel-synthetics.readthedocs.io/en/stable/)
- [Configuration](https://gretel-synthetics.readthedocs.io/en/stable/api/config.html)
- [Train your model](https://gretel-synthetics.readthedocs.io/en/stable/api/train.html)
- [Generate synthetic records](https://gretel-synthetics.readthedocs.io/en/stable/api/generate.html)

## Try it out now

If you want to quickly discover gretel-synthetics, simply click the button below and follow the tutorials!

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gretelai/gretel-synthetics/blob/master/examples/synthetic_records.ipynb)

Check out additional examples [here](https://github.com/gretelai/gretel-synthetics/tree/master/examples).

## Getting Started

This section will guide you through installation of `gretel-synthetics` and dependencies that are not directly installed by the Python package manager.

### Dependency Requirements

By default, we do not install certain core requirements, the following dependencies should be installed _external to the installation_
of `gretel-synthetics`, depending on which model(s) you plan to use.

- Torch: Used by Timeseries DGAN and ACTGAN (for ACTGAN, Torch is installed by SDV), we recommend version 2.0
- SDV (Synthetic Data Vault): Used by ACTGAN, we recommend version 0.17.x

These dependencies can be installed by doing the following:

```
pip install sdv<0.18 # for ACTGAN
pip install torch==2.0 # for Timeseries DGAN
```

To install the actual `gretel-synthetics` package, first clone the repo and then...

```
pip install -U .
```

_or_

```
pip install gretel-synthetics
```

_then..._

```
pip install jupyter
jupyter notebook
```

When the UI launches in your browser, navigate to `examples/synthetic_records.ipynb` and get generating!

If you want to install `gretel-synthetics` locally and use a GPU (recommended):

1. Create a virtual environment (e.g. using `conda`)

```
conda create --name tf python=3.9
```

2. Activate the virtual environment

```
conda activate tf
```

3. Run the setup script `./setup-utils/setup-gretel-synthetics-tensorflow24-with-gpu.sh`

The last step will install all the necessary software packages for GPU usage, `tensorflow=2.8` and `gretel-synthetics`.
Note that this script works only for Ubuntu 18.04. You might need to modify it for other OS versions.

## Timeseries DGAN Overview

The [timeseries DGAN module](https://synthetics.docs.gretel.ai/en/stable/models/timeseries_dgan.html#timeseries-dgan) contains a PyTorch implementation of a DoppelGANger model that is optimized for timeseries data. Similar to tensorflow, you will need to manually install pytorch:

```
pip install torch==1.13.1
```

[This notebook](https://github.com/gretelai/gretel-synthetics/blob/master/examples/timeseries_dgan.ipynb) shows basic usage on a small data set of smart home sensor readings.

## ACTGAN Overview

ACTGAN (Anyway CTGAN) is an extension of the popular [CTGAN implementation](https://sdv.dev/SDV/user_guides/single_table/ctgan.html) that provides
some additional functionality to improve memory usage, autodetection and transformation of columns, and more.

To use this model, you will need to manually install SDV:

```
pip install sdv<0.18
```

Keep in mind that this will also install several dependencies like PyTorch that SDV relies on, which may conflict with PyTorch
versions installed for use with other models like Timeseries DGAN.

The ACTGAN interface is a superset of the CTGAN interface. To see the additional features, please take a look at the ACTGAN demo notebook in the `examples` directory of this repo.
