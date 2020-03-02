# gretel-synthetics

This code has been developed and tested on Python 3.7. Python 3.8 is currently unsupported.

This package allows developers to quickly get emersed with synthetic data generation through the use of neural networks. The more complex pieces of working with libraries like Tensorflow and differential privacy are bundled into friendly Python classes and functions.

For example usage, please launch the example Jupyter Notebook and step through the config, train, and generation examples.

**NOTE**: The settings in the Jupyter Notebook are optimized to run on a CPU, so you can get the hang of how things work. We 
highly recommend running with no `max_char` limitation and at least 30 epochs on a GPU.


# Getting Started

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
