# -*- coding: utf-8 -*-
"""
CAUTION: Work in progress! The API (and this example) is not yet stable and may change without deprecation. Please reach out if you want to use or contribute to this library.

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

# Example dataset X with 3 samples and 5 features, all bool-encoded binary features.
X = np.array(
    [
        [1, 1, 0, 0, 1],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 0, 0],
    ],
).astype(bool)

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

# %%
# DataFrame + DiGraph: the one-call preprocessing path
# ----------------------------------------------------
# ``HierarchicalPreprocessor`` accepts a pandas ``DataFrame`` together with a
# named ``networkx.DiGraph`` hierarchy. The column-to-node mapping is then
# derived automatically from the DataFrame's column names, so neither
# ``nx.to_numpy_array`` nor an explicit ``columns`` argument is needed.

import pandas as pd

from scihfs.preprocessing import HierarchicalPreprocessor

# Bool DataFrame: each column is a (leaf) feature named after a hierarchy node.
df = pd.DataFrame(
    {
        "dog": [True, False, False],
        "cat": [False, True, False],
        "eagle": [False, False, True],
    }
)

# Named hierarchy. Nodes carry meaningful labels instead of column indices.
graph = nx.DiGraph(
    [
        ("animal", "mammal"),
        ("animal", "bird"),
        ("mammal", "dog"),
        ("mammal", "cat"),
        ("bird", "eagle"),
    ]
)

# One call: no nx.to_numpy_array, no manual column mapping, no columns=.
preprocessor = HierarchicalPreprocessor(graph)

# Opt into DataFrame output to keep the hierarchy node names on the columns.
preprocessor.set_output(transform="pandas")
preprocessor.fit(df)
df_transformed = preprocessor.transform(df)

print(df_transformed)

# %%
# DataFrame end-to-end: eager selection without a columns mapping
# ----------------------------------------------------------------
# The eager selectors accept the same DataFrame + named-DiGraph input, so the
# preprocessed frame can be fed straight into a selector -- again with no
# dedicated ``columns`` argument. With ``set_output(transform="pandas")`` (or
# any other supported dataframe library) the selected features keep their
# hierarchy node names.

selector = SHSELSelector(graph)
selector.set_output(transform="pandas")
selector.fit(df_transformed, y)

print(selector.get_feature_names_out())
print(selector.transform(df_transformed))
