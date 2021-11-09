Utils
=====

The utils module provides a number of different methods that are useful for training and working with synthetic data.

Some of these methods carry heavy dependencies such as scikit-learn. To prevent adding unnecessary requirements to the main gretel-synthetics package, util dependencies are shipped under an extra called, ``utils``. To install the ``utils`` extra, you may run


.. code-block:: shell

   pip install -U gretel-synthetics[utils]


.. toctree::
   :maxdepth: 2

   stats.rst
   header_clusters.rst
