#####################################
Getting Started with hfs
#####################################

Learn how to use
===================================================

1. Installation
-------------------------------------

The package cannot be installed with pip or conda yet so to create your package, you need to clone the ``hfs`` repository::

    $ git clone https://github.com/hasso-plattner-institute/scihfs.git

We recommend that you create a new virtual environment for hfs using poetry in which you install the required packages with::

    $ poetry install

2. Usage
-------------------------------------------
Here is a simple example of how to use one of the hierarchical feature selection algorithms implemented in hfs:

.. code-block:: python

    import networkx as nx
    import numpy as np

    from hfs import SHSELSelector
    from hfs.helpers import get_columns_for_numpy_hierarchy

    # Example dataset X with 3 samples and 5 features.
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
        ],
    )

    # Example labels
    y = np.array([1, 0, 0])

    # Example hierarchy graph : The node numbers refer to the dataset columns
    graph = nx.DiGraph([(0, 1), (1, 2), (2, 3), (0, 4)])

    # Create mapping from columns to hierarchy nodes
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])

    # Transform the hierarchy graph to a numpy array
    hierarchy = nx.to_numpy_array(hierarchy)

    # Initialize selector
    selector = SHSELSelector(hierarchy)

    # Fit selector and transform data
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)
