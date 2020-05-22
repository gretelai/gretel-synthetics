# Gretel Synthetics

![gretel-synthetics workflows](https://github.com/gretelai/gretel-synthetics/workflows/gretel-synthetics%20workflows/badge.svg)

[![Documentation Status](https://readthedocs.org/projects/gretel-synthetics/badge/?version=stable)](https://gretel-synthetics.readthedocs.io/en/stable/?badge=stable)


This code has been developed and tested on Python 3.7. Python 3.8 is currently unsupported. While not developed on Python 3.6, this code will run in Google Colab, which currently uses 3.6. If you wish to use Python 3.6, out side of Google Colab, you may install with the `3.6` extras: `pip install gretel-synthetics[tf,3.6]`, for example.

This package allows developers to quickly get immersed with synthetic data generation through the use of neural networks. The more complex pieces of working with libraries like Tensorflow and differential privacy are bundled into friendly Python classes and functions.

For example usage, please launch the example Jupyter Notebook and step through the config, train, and generation examples.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gretelai/gretel-synthetics/blob/master/examples/synthetic_records.ipynb)

**NOTE**: The settings in our Jupyter Notebook examples are optimized to run on a GPU, which you can experiment with
for free in Google Colaboratory. If you're running on a CPU, you might want to grab a cup of coffee, 
or lower `max_lines` and `epochs` to 5000 and 10, respectively.


# Getting Started
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
