# Gretel Synthetics
<p align="center">
    <a href="https://gretel.ai"><img width="128px" src="https://gretel-public-website.s3.amazonaws.com/assets/gobs_the_cat_@1x.png" alt="Gobs the Gretel.ai cat" /></a><br />
    <i>An open source synthetic data library from Gretel.ai</i>
</p>

![gretel-synthetics workflows](https://github.com/gretelai/gretel-synthetics/workflows/gretel-synthetics%20workflows/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/gretel-synthetics/badge/?version=stable)](https://gretel-synthetics.readthedocs.io/en/stable/?badge=stable)
![GitHub](https://img.shields.io/github/license/gretelai/gretel-synthetics)
[![PyPI](https://badge.fury.io/py/gretel-synthetics.svg)](https://badge.fury.io/py/gretel-synthetics)
[![Python](https://img.shields.io/pypi/pyversions/gretel-synthetics.svg)](https://github.com/gretelai/gretel-synthetics)
[![Downloads](https://pepy.tech/badge/gretel-synthetics)](https://pepy.tech/project/gretel-synthetics)
[![GitHub stars](https://img.shields.io/github/stars/gretelai/gretel-synthetics?style=social)](https://github.com/gretelai/gretel-synthetics)

## Documentation
* [Get started with gretel-synthetics](https://gretel-synthetics.readthedocs.io/en/stable/)
* [Configuration](https://gretel-synthetics.readthedocs.io/en/stable/api/config.html)
* [Train your model](https://gretel-synthetics.readthedocs.io/en/stable/api/train.html)
* [Generate synthetic recoreds](https://gretel-synthetics.readthedocs.io/en/stable/api/generate.html)

## Overview

This package allows developers to quickly get immersed with synthetic data generation through the use of neural networks. The more complex pieces of working with libraries like Tensorflow and differential privacy are bundled into friendly Python classes and functions.


**NOTE**: The settings in our Jupyter Notebook examples are optimized to run on a GPU, which you can experiment with
for free in Google Colaboratory. If you're running on a CPU, you might want to grab a cup of coffee, 
or lower `max_lines` and `epochs` to 5000 and 10, respectively. This code is developed for TensorFlow 2.3.X and above.


## Try it out now!
If you want to quickly discover gretel-synthetics, simply click the button below and follow the tutorials!

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gretelai/gretel-synthetics/blob/master/examples/synthetic_records.ipynb)

Check out additional examples [here](https://github.com/gretelai/gretel-synthetics/tree/master/examples).

## Getting Started

By default, we do not install Tensorflow via pip as many developers and cloud services such as Google Colab are
running customized versions for their hardware. If you wish to pip install Tensorflow along with gretel-synthetics,
use the [tf] commands below instead.

```
pip install -U .                     # Do not install Tensorflow by default (assuming you have built a distro for your hardware)
pip install -U -e ".[tf]"            # Install a pinned version of Tensorflow"
```

_or_

```
pip install gretel-synthetics        # Do not install Tensorflow by default (assuming you have built a distro for your hardware)
pip install gretel-synthetics[tf]    # Install a pinned version of Tensorflow
```

_then..._

```
$ pip install jupyter
$ jupyter notebook
```

When the UI launches in your browser, navigate to `examples/synthetic_records.ipynb` and get generating!

