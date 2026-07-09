#####################################
Getting Started with scihfs
#####################################

Learn how to use
===================================================

1. Installation
-------------------------------------

The package cannot be installed with pip or conda yet so to create your package, you need to clone the ``scihfs`` repository::

    $ git clone https://github.com/hasso-plattner-institute/scihfs.git

We recommend that you create a new virtual environment for scihfs in which you install the required packages with::

    $ uv sync

2. Usage
-------------------------------------------
Here is a simple example of how to use one of the hierarchical feature selection algorithms implemented in scihfs:

.. code-block:: python

    import networkx as nx
    import numpy as np

    from scihfs import SHSELSelector
    from scihfs.helpers import get_columns_for_numpy_hierarchy

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
    columns = get_columns_for_numpy_hierarchy(graph, X.shape[1])

    # Transform the hierarchy graph to a numpy array
    hierarchy = nx.to_numpy_array(graph)

    # Initialize selector
    selector = SHSELSelector(hierarchy)

    # Fit selector and transform data
    selector.fit(X, y, columns=columns)
    X_transformed = selector.transform(X)

If your data lives in a pandas ``DataFrame`` whose column names match the
node names of a ``networkx.DiGraph`` hierarchy, the mapping steps disappear:
Pass the graph and the DataFrame directly and the column-to-node mapping is
derived automatically. With ``set_output(transform="pandas")`` the selected
features keep their names:

.. code-block:: python

    selector = SHSELSelector(graph)
    selector.set_output(transform="pandas")
    selector.fit(df, y)
    df_transformed = selector.transform(df)

The DataFrame workflow is currently only supported by the
``HierarchicalPreprocessor`` and the eager selectors; the lazy selectors are
arrays-only.
