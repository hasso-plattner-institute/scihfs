# -*- coding: utf-8 -*-
"""
Eager learning
=====================

These examples show you how to use the library's hierarchical feature selection methods for eager learning.
In this example the SHSELSelector is used. However, you can replace this class with any other eager
hierarchical feature selector class and use it in exactly the same way.
"""

# %%
# Artificial data
# ----------------
# This is just a simple example using artificial data to show you how the
# Selector is used. In this example only the feature selection step and no
# classification is performed.

import networkx as nx
import numpy as np

from scihfs.helpers import get_columns_for_numpy_hierarchy
from scihfs.selectors import SHSELSelector

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

print(X_transformed)
