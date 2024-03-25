====================================================
hfs - A library for hierarchical feature selection
====================================================

Introduction
=============

Welcome to the **hfs** repository!👋 
This library provides several hierarchical feature selection algorithms.

Many real-world settings contain hierarchical relations. While in text mining, words can be ordered in generalization-specialization relationships in bioinformatics the function of genes is often described as a hierarchy. We can make use of these relationships between datasets' features by using special hierarchical feature selection algorithms that reduce redundancy in the data. This can not only make tasks like classification faster but also improve the results. Depending on use case and preference you can choose from lazy and eager hierarchical feature selection algorithms in this library.

Getting Started
===================================================

1. Installation
-------------------------------------

The package cannot be installed with pip or conda yet so to create your package, you need to clone the ``hfs`` repository::

    ``git clone https://github.com/hasso-plattner-institute/hfs.git

    Install the environment using::

    ```poetry install```

1. Usage
-------------------------------------------
Here is a simple example of how to use one of the hierarchical feature selection algorithms implemented in hfs:

.. code-block:: python

    from hfs import SHSELSelector
    
    # Initialize selector
    selector = SHSELSelector(hierarchy)

    # Fit selector and transform data
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)

Documentation
=============

For detailed information on how to use **hfs**, check out our complete documentation at https://hfs.readthedocs.io. 📖

There you can find not only the API documentation but also more examples, background information on the algorithms we implemented and results for some experiments we performed with them.

Contributing
============

We welcome contributions! If you would like to contribute to the project, 
feel free to create a pull request.

Linting and Testing
```
poetry run black .
```

```
poetry run pytest hfs
```
Happy feature selecting!
