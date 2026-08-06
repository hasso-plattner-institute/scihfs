# -*- coding: utf-8 -*-
"""
CAUTION: Work in progress! The API (and this example) is not yet stable and may change without deprecation. Please reach out if you want to use or contribute to this library.

Lazy learning
=====================

These examples show how to use the library's *lazy* hierarchical feature selection
methods -- which are embedded methods that (a) behave like a classifier, and (b)
are not trained in the usual sense, but apply their selection logic per test instance.

Following the scikit-learn *classifiers* scheme: ``fit`` does the
one-off work (such as determining relevance scores for all features in the dataset),
while ``predict`` selects a feature subset *per test instance* AND at the same time
classifies it.
[``select`` exposes those per-instance selection masks on their own.]
Any selector below can be swapped for another and used in exactly the same way.
"""

import networkx as nx
import numpy as np

from scihfs.metrics import mean_selected_fraction, sensitivity_specificity_product
from scihfs.selectors import HIP, HNB, MR, RNB, TAN, HNBs

# 4 training + 2 test samples, 4 bool features.
X_train = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]], dtype=bool)
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool)
y_test = np.array([0, 1])

# Hierarchy over the 4 columns (node i corresponds to column i), as an
# adjacency matrix -- the positional column<->node default applies.
hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3)]))

model = HNB(hierarchy=hierarchy, k=3)
model.fit(X_train, y_train)

# ``predict`` -> per-instance labels; ``select`` -> per-instance selection masks.
print("predictions:", model.predict(X_test))
print("selection masks:\n", model.select(X_test))

# Both from a single sweep -- ``masks`` is identical to ``select(X_test)``.
predictions, masks = model.predict(X_test, return_masks=True)

# Calibrated class probabilities and the built-in accuracy score.
print("probabilities:\n", model.predict_proba(X_test))
print("accuracy:", model.score(X_test, y_test))

# The bespoke metrics that are used in many of the lazy HFS papers.
print("sensitivity*specificity:", sensitivity_specificity_product(y_test, predictions))
print("mean selected fraction:", mean_selected_fraction(masks))

# Lazy HFS methods have the same API and can be used interchangeably.
family = [
    ("HIP", HIP(hierarchy=hierarchy)),
    ("HNB", HNB(hierarchy=hierarchy, k=3)),
    ("HNBs", HNBs(hierarchy=hierarchy)),
    ("RNB", RNB(hierarchy=hierarchy, k=3)),
    ("MR", MR(hierarchy=hierarchy)),
    ("TAN", TAN(hierarchy=hierarchy)),
]
for name, selector in family:
    selector.fit(X_train, y_train)
    print(f"{name}: {selector.predict(X_test)}")

# %%
# DataFrame input: the column->node mapping is auto-derived
# ---------------------------------------------------------
# All scihfs methods also accept a pandas ``DataFrame`` as input, in addition
# to plain arrays and sparse matrices.
# If combined with a named ``networkx.DiGraph`` hierarchy, the column->node mapping is auto-derived from the feature names, so that no explicit ``columns=`` argument is needed.

import pandas as pd

# Bool DataFrame whose columns are named after hierarchy nodes, in shuffled
# order (so a positional mapping would be wrong).
df_train = pd.DataFrame(
    {
        "cat": [1, 0, 1, 1],
        "animal": [1, 1, 1, 1],
        "eagle": [0, 0, 1, 0],
        "mammal": [1, 0, 1, 1],
    }
).astype(bool)
df_test = pd.DataFrame(
    {
        "cat": [0, 1],
        "animal": [1, 1],
        "eagle": [0, 1],
        "mammal": [1, 1],
    }
).astype(bool)

# Named hierarchy -- nodes carry meaningful labels instead of column indices.
named_hierarchy = nx.DiGraph(
    [("animal", "mammal"), ("mammal", "cat"), ("animal", "eagle")]
)

model = HNB(hierarchy=named_hierarchy, k=3)
model.fit(df_train, y_train)

# The mapping was derived from the feature names (columns cat, animal, eagle,
# mammal -> nodes 2, 0, 3, 1).
print("auto-derived column->node mapping:", model.get_columns())
print("predictions:", model.predict(df_test))
print("selection masks (plain ndarray):\n", model.select(df_test))

# %%
# Sparse input: Input data and masks stay sparse
# --------------------------------------------------------
# Lazy's ``fit`` and ``predict``/``select`` all accept sparse input and keep it sparse
# throughout their internal processing.

import scipy.sparse as sp

model = HNB(hierarchy=hierarchy, k=3).fit(sp.csr_array(X_train), y_train)
sparse_masks = model.select(sp.csr_array(X_test))

print("predictions:", model.predict(sp.csr_array(X_test)))
print("selection masks (sparse):", repr(sparse_masks))
# The metrics take the sparse masks directly, without densifying them.
print("mean selected fraction:", mean_selected_fraction(sparse_masks))

# Dense input still yields a plain ndarray.
print("same selection, dense input:\n", model.select(X_test))
