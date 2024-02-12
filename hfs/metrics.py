"""
Different metric functions.
"""
from collections import Counter

import numpy as np
from info_gain.info_gain import info_gain, info_gain_ratio
from numpy.linalg import norm
from scipy import sparse

from hfs.lib.pyitlib import information_mutual_conditional as imc


def lift(data, labels):
    """Calculates the lift value for each feature in the data.

    Parameters
    ----------
    data : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    labels : array-like, shape (n_samples,)
        The target values. An array of int. Not needed for all estimators.

    Returns
    ----------
    lift_values : list, length n_features
                The lift values for all features. List of floats.
    """
    lift_values = []
    num_samples, num_features = data.shape

    for index in range(num_features):
        # deal with sparse matrices
        if sparse.issparse(data):
            data = data.tocsr()
            column = data[:, index]
            non_zero_values = column.size
        else:
            column = data[:, index]
            non_zero_values = np.count_nonzero(column)

        prob_feature = non_zero_values / num_samples

        if non_zero_values > 0:
            prob_event_conditional = (
                len(
                    [
                        value
                        for index, value in enumerate(column)
                        if value != 0 and labels[index] != 0
                    ]
                )
                / non_zero_values
            )

            lift_values.append(prob_event_conditional / prob_feature)
        else:
            lift_values.append(0)
    return lift_values


def information_gain(data, labels):
    """Calculates the information gain for each feature in the data.

    Parameters
    ----------
    data : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    labels : array-like, shape (n_samples,)
        The target values. An array of int. Not needed for all estimators.

    Returns
    ----------
    ig_values : list, length n_features
                The information gain values for all features.
                List of floats.
    """
    ig_values = []
    for column_index in range(data.shape[1]):
        ig = info_gain(data[:, column_index], labels)
        ig_values.append(ig)
    return ig_values


def conditional_mutual_information(node1, node2, y):
    """Calculates conditional mutual information for two features using the dit library.

    Parameters
    ----------
    node1 : numpy.ndarray, shape (n_samples,)
            All values from the training set for one feature.
    node2 : numpy.ndarray, shape (n_samples,)
            All values from the training set for another feature.
    y : numpy.ndarray, shape (n_samples,)
            The target values. An array of int. Not needed for all estimators.

    Returns
    ----------
    float : The conditional mutual information value.
    """
    return imc(node1, node2, y)


def cosine_similarity(i: np.ndarray, j: np.ndarray):
    """Calculates the cosine similarity for two rows from the dataset.

    Parameters
    ----------
    i : numpy.ndarray, shape (n_features,)
        All features for one sample from the dataset.
    j : numpy.ndarray, shape (n_features,)
        All features for another sample from the dataset.

    Returns
    ----------
    float : The cosine similarity for the input rows.
    """
    return np.dot(i, j) / (norm(i) * norm(j))


def gain_ratio(data, labels):
    """Calculates the information gain ratio for each feature.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data samples.
    y : array-like, shape (n_samples,)
        The target values. An array of int.

    Returns
    ----------
    gr_values : list, length n_features
                A list of floats containing the information gain
                values for each feature in the dataset.
    """
    gr_values = []
    for column_index in range(data.shape[1]):
        gr = info_gain_ratio(data[:, column_index], labels)
        gr_values.append(gr)
    return gr_values


def pearson_correlation(i: np.ndarray, j: np.ndarray):
    """Calculates the correlation between two vectors.

    Parameters
    ----------
    i : {array-like, sparse matrix}, shape (n_samples,)
        One feature vector.
    j : {array-like, sparse matrix}, shape (n_samples,)
        Another feature vector.

    Returns
    ----------
    float : The pearson correlation between the input vectors.
    """
    return np.corrcoef(i, j)[0, 1]
