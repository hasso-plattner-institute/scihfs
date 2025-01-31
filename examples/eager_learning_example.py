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

from hfs.helpers import get_columns_for_numpy_hierarchy
from hfs.selectors import SHSELSelector

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


# %%
# Real data
# -----------
# This is an example using a real world dataset. The SportsTweets Dataset consists
# of hierarchical type features that were extracted from tweets. These tweets are
# classified as either a sports tweet or not a sports tweet. The classification step
# is performed with a Naive Bayes Classifier implemented in Scikit-learn.
#

import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

from hfs.data_utils import create_mapping_columns_to_nodes, load_data, process_data
from hfs.preprocessing import HierarchicalPreprocessor
from hfs.selectors.shsel import SHSELSelector


# Preprocess hierarchy and dataset before feature selection to ensure all nodes
# in the hierarchy have a corresponding node in the dataset and the other way around.
def preprocess_data(hierarchy, X_train, X_test, columns):
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X_train, columns=columns)
    X_train_transformed = preprocessor.transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    hierarchy_updated = preprocessor.get_hierarchy()
    columns_updated = preprocessor.get_columns()
    return X_train_transformed, X_test_transformed, hierarchy_updated, columns_updated


# Load the tweet dataset and split it into a train and a test set.
def get_tweet_data():
    X, y, hierarchy = load_data(test_version=True)
    columns = create_mapping_columns_to_nodes(X, hierarchy)
    X = X.to_numpy()
    hierarchy = nx.to_numpy_array(hierarchy)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0
    )
    return hierarchy, X_train, y_train, X_test, y_test, columns


# If you have not done this already call process_data once to preprocess the dataset and create
# the hierarchy graph.
process_data(test_version=True)

# Load the data and preprocess it.
hierarchy, X_train, y_train, X_test, y_test, columns = get_tweet_data()
X_train, X_test, hierarchy, columns = preprocess_data(hierarchy, X_train, X_test, columns)

# Initialize the Selector, fit it on the training data and transform test and training data.
selector = SHSELSelector(hierarchy)
selector.fit(X_train, y_train, columns=columns)
X_transformed = selector.transform(X_train)
X_test_transformed = selector.transform(X_test)

# Train a classifier on the transformed training dataset and evaluate it with
# the transformed test dataset.
classifier = BernoulliNB()
classifier.fit(X_transformed, y_train)
accuracy = classifier.score(X_test_transformed, y_test)

print(accuracy)
