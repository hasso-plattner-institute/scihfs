
**IMPORTANT: This repository is under active development. Prior to the first release version, all contents are still preliminary and might change without notice.**

[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

====================================================
scihfs - A library for hierarchical feature selection
====================================================

Introduction
=============

Welcome to the **scihfs** repository!👋
This library provides several hierarchical feature selection algorithms.

Many real-world settings contain hierarchical relations. While in text mining, words can be ordered in generalization-specialization relationships in bioinformatics the function of genes is often described as a hierarchy. We can make use of these relationships between datasets' features by using special hierarchical feature selection algorithms that reduce redundancy in the data. This can not only make tasks like classification faster but also improve the results. Depending on use case and preference you can choose from lazy and eager hierarchical feature selection algorithms in this library.

Getting Started
===================================================

1. Installation
-------------------------------------

The package cannot be installed with pip or conda yet so to create your package, you need to clone the ``scihfs`` repository::

    ``git clone https://github.com/hasso-plattner-institute/scihfs.git

    Install the environment using::

    ```uv sync```

1. Usage
-------------------------------------------
Here is a simple example of how to use one of the hierarchical feature selection algorithms implemented in hfs:

.. code-block:: python

    from scihfs import SHSELSelector

    # hierarchy: a networkx.DiGraph whose node names match the columns of the
    # DataFrame X. (Adjacency matrices and plain arrays are supported too --
    # then pass an explicit columns= mapping to fit.)
    selector = SHSELSelector(hierarchy)

    # Fit selector and transform data
    selector.fit(X, y)
    X_transformed = selector.transform(X)

Documentation
=============

For detailed information on how to use **scihfs**, check out our complete documentation at https://scihfs.readthedocs.io. 📖

There you can find not only the API documentation but also more examples, background information on the algorithms we implemented and results for some experiments we performed with them.

Contributing
============

We welcome contributions! If you would like to contribute to the project,
feel free to create a pull request.

Linting and Testing
```
uv run black .
```

```
uv run pytest scihfs
```

Pre-Commit Hooks
To run the pre-commit hooks, you can use the following command:
```
pre-commit run --all-files
```

Happy feature selecting!🌟
