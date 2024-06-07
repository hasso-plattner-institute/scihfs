import networkx as nx
import numpy as np

from hfs.helpers import get_columns_for_numpy_hierarchy
from hfs import GASel

seed_value = 40
np.random.seed(seed_value)

# Example : synthetic dataset
X = np.random.randint(2, size=(30, 7))
y = np.random.randint(2, size=30)

print("Data:")
print(X)
print(y)

# Example : synthetic hierarchy graph, the node numbers refer to the dataset columns
graph = nx.DiGraph(
    [(0, 1), (0, 2), (1, 3), (1, 6), (2, 4), (2, 5), (2, 6), (3, 5), (0, 6)]
)

# Create mapping from columns to hierarchy nodes
columns = get_columns_for_numpy_hierarchy(graph, X.shape[1])

# Transform the hierarchy graph to a numpy array
hierarchy = nx.to_numpy_array(graph, nodelist=[0, 1, 2, 3, 4, 5, 6])


# Initialize selector
she_selector = GASel(hierarchy, mode="she")

# Fit selector and transform data
she_selector.fit(X, y, columns=columns)

print(she_selector.selected_features_)
