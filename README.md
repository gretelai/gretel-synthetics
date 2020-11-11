# Gretel Synthetics

![gretel-synthetics workflows](https://github.com/gretelai/gretel-synthetics/workflows/gretel-synthetics%20workflows/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/gretel-synthetics/badge/?version=stable)](https://gretel-synthetics.readthedocs.io/en/stable/?badge=stable)
[![PyPI](https://badge.fury.io/py/gretel-synthetics.svg)](https://badge.fury.io/py/gretel-synthetics)
[![Python](https://img.shields.io/pypi/pyversions/tensorflow.svg?style=plastic)](https://badge.fury.io/py/gretel-synthetics)
[![Downloads](https://pepy.tech/badge/gretel-synthetics)](https://pepy.tech/project/gretel-synthetics)
[![GitHub stars](https://img.shields.io/github/stars/gretelai/gretel-synthetics?style=social)](https://github.com/gretelai/gretel-synthetics)

**`Documentation`** |
------------------- |
[![Documentation](https://img.shields.io/badge/api-reference-blue.svg)](https://gretel-synthetics.readthedocs.io/en/stable/) |

# Overview

This package allows developers to quickly get immersed with synthetic data generation through the use of neural networks. The more complex pieces of working with libraries like Tensorflow and differential privacy are bundled into friendly Python classes and functions.

For example usage, please launch the example Jupyter Notebook and step through the config, train, and generation examples.
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gretelai/gretel-synthetics/blob/master/examples/synthetic_records.ipynb)

**NOTE**: The settings in our Jupyter Notebook examples are optimized to run on a GPU, which you can experiment with
for free in Google Colaboratory. If you're running on a CPU, you might want to grab a cup of coffee, 
or lower `max_lines` and `epochs` to 5000 and 10, respectively. This code is developed for TensorFlow 2.3.X and above.

# Roadmap
 
## Pre 0.14.X
 
Prior to the 0.14.x versions of Gretel Synthetics, we noticed that the differential privacy library we are using (tensorflow-privacy) may not be properly called based on the version of TensorFlow being used, particularly TF 2.1+. What this means is that with the `dp` option enabled on versions before 0.14.X, the synthetic data may not have been run through DP optimizers properly. We are currently working with the TensorFlow privacy team on an update to resolve this situation.
 
## 0.14.X
 
This release series will continue to operate as the versions prior and we will continue to add new functionality that makes training more automated and user friendly. Some enhancements are incorporating Keras' features to do early stopping of model training based on observed loss or accuracy and ensuring that the best versions of models are stored.  This will remove the need to guess an optimal number of training epochs and help train the best model sooner.
 
One temporary change that will be done in this release series is throwing a `RuntimeError` in the event the `dp` option is enabled. We are doing this for a couple of reasons:
 
1) We want to reduce the risk DP is not applied properly to your data.  By default, `dp` has always been disabled by default, so this will continue to remain the case.
 
2) We did not want to drastically change the signature of the configuration object. By removing these options it becomes more ambiguous to throw a `TypeError` because of removed parameters than it does to throw a `RunTimeError` with a more detailed explanation of why the option cannot be used temporarily.
 
 
## 0.15.X
 
We are currently working to ensure that our differentially private optimizers are called correctly when enabled, and plan to introduce them in this release series. To correctly subclass the standard non-differentially private optimizers in a future-proof way, we are leveraging the Keras V2 optimizer interfaces introduced in TensorFlow 2.4.x. Additionally, we will be doing a significant amount of hyperparameter optimization and provide default optimizers and hyperparameters for non-DP and DP training. 

In this release you may expect to see an interface change to the configuration object. We are exploring the use of an `optimizer` parameter that will take an optional `Optimizer()` or `DPOptimizer()` class that you can instantiate yourself and provide to the configuration. This will allow you to explore multiple optimizers with your data. We will still continue to provide the `dp` boolean option that if used will default to optimal `Optimizer()` or `DPOptimizer()` objects based on our hyperparameter testing and should work well for a variety of general synthetic use cases.


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
