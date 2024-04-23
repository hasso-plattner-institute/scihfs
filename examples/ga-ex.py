import networkx as nx
import numpy as np

from hfs.helpers import get_columns_for_numpy_hierarchy
from hfs import gasel

"""
# Synthetic dataset with 5 features and 20 elements
X = np.array([
    [0, 0, 1, 1, 0],
    [0, 0, 1, 1, 0],
    [0, 1, 1, 1, 1],
    [1, 0, 1, 1, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 1, 1],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 1, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 0, 1],
    [0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0],
    [1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1],
    [0, 0, 1, 1, 0],
    [1, 1, 0, 1, 1],
    [0, 0, 0, 0, 1],
    [0, 1, 0, 0, 1],
])

# Synthetic labels for the dataset
y = np.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1])
"""


X = np.random.randint(2, size=(30, 7))
y = np.random.randint(2, size=30)
# Example hierarchy graph : The node numbers refer to the dataset columns
graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (0, 6)])

# Create mapping from columns to hierarchy nodes
columns = get_columns_for_numpy_hierarchy(graph, X.shape[1])

# Transform the hierarchy graph to a numpy array
hierarchy = nx.to_numpy_array(graph)

# Initialize selector
selector = gasel(hierarchy)

# Fit selector and transform data
selector.fit(X, y, columns=columns)
X_transformed = selector.transform(X)

print(X_transformed)
print(selector.get_feature_support())
