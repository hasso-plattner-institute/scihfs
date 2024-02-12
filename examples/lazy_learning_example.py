# -*- coding: utf-8 -*-
# %%
"""
Lazy learning
=====================

These methods can help you with the correct usage of the library's lazy learning methods.
The dataset as well as the hierarchy are structures not following a specific pattern, but just showing how to use the interfaces properly.
"""

import networkx as nx
import numpy as np

from hfs.preprocessing import HierarchicalPreprocessor
from hfs.selectors import HIP, HNB, MR, RNB, TAN, HieAODE, HNBs


# Define data
def data():
    train_x_data = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    train_y_data = np.array([0, 0, 1, 1])
    test_x_data = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
    test_y_data = np.array([0, 1])
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3)]))
    return (train_x_data, train_y_data, test_x_data, test_y_data, hierarchy)


# Preprocess to fit data and hierarchy to each other
def preprocess():
    train_x_data, train_y_data, test_x_data, test_y_data, hierarchy = data()
    preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
    preprocessor.fit(train_x_data)
    train = preprocessor.transform(train_x_data)
    test = preprocessor.transform(test_x_data)
    hierarchy = preprocessor.get_hierarchy()
    return (train, test, train_y_data, test_y_data, hierarchy)


train, test, train_y_data, test_y_data, hierarchy = preprocess()
# %%
"""
=========================================================================
HieAODE - 
=========================================================================
"""

print("\nHieAODE:")
# Initialize and fit HNB model with threshold k = 3 features to select
model = HieAODE(hierarchy=hierarchy)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)
# %%
# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)
# %%
# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)

"""
=========================================================================
HNB - Hierarchy Based Redundant Attribute Removal Naive Bayes Classifier
=========================================================================
"""

print("\nHNB:")
# Initialize and fit HNB model with threshold k = 3 features to select
model = HNB(hierarchy=hierarchy, k=3)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)

# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)

# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)

# %%
"""
=========================================================================
HNB-s 
=========================================================================
"""

print("HNB-s:")
# Initialize and fit HNBs model
model = HNBs(hierarchy=hierarchy)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)

# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)

# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)


"""
==================================
RNB - Relevance-based Naive Bayes
==================================
"""

print("\nRNB:")
# Initialize and fit RNB model with threshold k = 3 features to select
model = RNB(hierarchy=hierarchy)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)

# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)

# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)


"""
=======================================
MR -  Most Relevant Feature Selection
=======================================
"""
print("\nMR:")
# Initialize and fit MR model
model = MR(hierarchy=hierarchy)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)

# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)

# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)


"""
==========================================
HIP - Hierarchical Information Preserving
==========================================
"""
print("\nHIP:")
# Initialize and fit HIP model
model = HIP(hierarchy=hierarchy)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)

# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)

# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)

"""
====================================================================
TAN - Hierarchical Redundancy Eliminated Tree Augmented Naive Bayes
====================================================================
"""
print("\nTAN:")
# Initialize and fit Tan model
model = TAN(hierarchy=hierarchy)
model.fit_selector(X_train=train, y_train=train_y_data, X_test=test)

# Select features and predict
predictions = model.select_and_predict(predict=True, saveFeatures=True)
print(predictions)

# Calculate score
score = model.get_score(test_y_data, predictions)
print(score)
