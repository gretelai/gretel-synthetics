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

## Try it out now!

If you want to quickly discover gretel-synthetics, simply click the button below and follow the tutorials!

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gretelai/gretel-synthetics/blob/master/examples/synthetic_records.ipynb)

Check out additional examples [here](https://github.com/gretelai/gretel-synthetics/tree/master/examples).

## Getting Started

This section will guide you through installation of `gretel-synthetics` and dependencies that are not directly installed by the Python package manager.

### Dependency Requirements

By default, we do not install certain core requirements, the following dependencies should be installed _external to the installation_
of `gretel-synthetics`, depending on which model(s) you plan to use.

- Tensorflow: Used by the LSTM model, we recommend version 2.11.x
- Torch: Used by Timeseries DGAN and ACTGAN (for ACTGAN, Torch is installed by SDV), we recommend version 2.0
- SDV (Synthetic Data Vault): Used by ACTGAN, we recommend version 0.17.x

These dependencies can be installed by doing the following:

```
pip install tensorflow==2.11 # for LSTM
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
$ pip install jupyter
$ jupyter notebook
```

When the UI launches in your browser, navigate to `examples/synthetic_records.ipynb` and get generating!

If you want to install `gretel-synthetics` locally and use a GPU (recommended):

1. Create a virtual environment (e.g. using `conda`)

```
$ conda create --name tf python=3.9
```

2. Activate the virtual environment

```
$ conda activate tf
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

## LSTM Overview

This package allows developers to quickly get immersed with synthetic data generation through the use of neural networks. The more complex pieces of working with libraries like Tensorflow and differential privacy are bundled into friendly Python classes and functions. There are two high level modes that can be utilized.

### Simple Mode

The simple mode will train line-per-line on an input file of text. When generating data, the generator will yield a custom object that can be used a variety of different ways based on your use case. [This notebook](https://github.com/gretelai/gretel-synthetics/blob/master/examples/tensorflow/simple-character-model.ipynb) demonstrates this mode.

### DataFrame Mode

This library supports CSV / DataFrames natively using the DataFrame "batch" mode. This module provided a wrapper around our simple mode that is geared for working with tabular data. Additionally, it is capable of handling a high number of columns by breaking the input DataFrame up into "batches" of columns and training a model on each batch. [This notebook](https://github.com/gretelai/gretel-synthetics/blob/master/examples/dataframe_batch.ipynb) shows an overview of using this library with DataFrames natively.

### Components

There are four primary components to be aware of when using this library.

1. Configurations. Configurations are classes that are specific to an underlying ML engine used to train and generate data. An example would be using `TensorFlowConfig` to create all the necessary parameters to train a model based on TF. `LocalConfig` is aliased to `TensorFlowConfig` for backwards compatibility with older versions of the library. A model is saved to a designated directory, which can optionally be archived and utilized later.

2. Tokenizers. Tokenizers convert input text into integer based IDs that are used by the underlying ML engine. These tokenizers can be created and sent to the training input. This is optional, and if no specific tokenizer is specified then a default one will be used. You can find [an example](https://github.com/gretelai/gretel-synthetics/blob/master/examples/tensorflow/batch-df-char-tokenizer.ipynb) here that uses a simple char-by-char tokenizer to build a model from an input CSV. When training in a non-differentially private mode, we suggest using the default `SentencePiece` tokenizer, an unsupervised tokenizer that learns subword units (e.g., **byte-pair-encoding (BPE)** [[Sennrich et al.](http://www.aclweb.org/anthology/P16-1162)]) and **unigram language model** [[Kudo.](https://arxiv.org/abs/1804.10959)]) for faster training and increased accuracy of the synthetic model.

3. Training. Training a model combines the configuration and tokenizer and builds a model, which is stored in the designated directory, that can be used to generate new records.

4. Generation. Once a model is trained, any number of new lines or records can be generated. Optionally, a record validator can be provided to ensure that the generated data meets any constraints that are necessary. See our notebooks for examples on validators.

### Utilities

In addition to the four primary components, the `gretel-synthetics` package also ships with a set of utilities that are helpful for training advanced synthetics models and evaluating synthetic datasets.

Some of this functionality carries large dependencies, so they are shipped as an extra called `utils`. To install these dependencies, you may run

```
pip install gretel-synthetics[utils]
```

For additional details, please refer to the [Utility module API docs](https://synthetics.docs.gretel.ai/en/latest/utils/index.html).

### Differential Privacy

Differential privacy support for our TensorFlow mode is built on the great work being done by the Google TF team and their [TensorFlow Privacy library](https://github.com/tensorflow/privacy).

When utilizing DP, we currently recommend using the character tokenizer as it will only create a vocabulary of single tokens and removes the risk of sensitive data being memorized as actual tokens that can be replayed during generation.

There are also a few configuration options that are notable such as:

- `predict_batch_size` should be set to 1
- `dp` should be enabled
- `learning_rate`, `dp_noise_multiplier`, `dp_l2_norm_clip`, and `dp_microbatches` can be adjusted to achieve various epsilon values.
- `reset_states` should be disabled

Please see our [example Notebook](https://github.com/gretelai/gretel-synthetics/blob/master/examples/tensorflow/diff_privacy.ipynb) for training a DP model based on the [Netflix Prize](https://en.wikipedia.org/wiki/Netflix_Prize) dataset.
