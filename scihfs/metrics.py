"""
Different metric functions.
"""

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from sklearn.metrics import mutual_info_score

from scihfs.pyitlib import information_mutual_conditional as imc


def _information_gain(feature: np.ndarray, target: np.ndarray) -> float:
    """Information gain (IG) of a ``feature`` (with respect to the ``target``), equivalent to the 'mutual information' (MI) and 'information measure' (IM).

    Def.: ``IG(feature; target) = H(target) - H(target | feature)``

    H(x) [H(x | y)] is the (conditional) entropy of x (given y).
    The terms "feature" / "attribute", and "target" / "class" are used interchangeably.
    """
    return mutual_info_score(feature, target)


def _gain_ratio(feature: np.ndarray, target: np.ndarray) -> float:
    """Gain ratio (GR) of a ``feature`` (with respect to the ``target``).

    Def.: ``GR(feature) = IG(feature; target) / H(feature)``

    H(x) is the entropy of x, equivalent to IG(x; x). IG is the information gain.
    The terms "feature" / "attribute", and "target" / "class" are used interchangeably.

    A constant feature carries no information and cannot be split
    (``H(feature) == 0``), so its gain ratio is defined as ``0``.
    """
    feature_entropy = mutual_info_score(feature, feature)
    if feature_entropy == 0:
        return 0.0
    return _information_gain(feature, target) / feature_entropy


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

    if sparse.issparse(data):
        # Ensure CSC format for subscriptability and efficient column access
        data = data.tocsc()

    labels = np.asarray(labels)  # Ensure labels are a NumPy array

    for index in range(num_features):
        if sparse.issparse(data):
            column = data[:, index]
            non_zero_mask = column.nonzero()[0]  # Indices of non-zero elements
            non_zero_values = len(non_zero_mask)
        else:
            column = data[:, index]
            non_zero_mask = column != 0
            non_zero_values = np.count_nonzero(column)

        prob_feature = non_zero_values / num_samples

        if non_zero_values > 0:
            prob_event_conditional = (
                np.count_nonzero(labels[non_zero_mask]) / non_zero_values
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

    Notes
    -----
    For the information gain definition, see :func:`_information_gain`.
    """
    ig_values = []
    if sparse.issparse(data):
        data = data.tocsc()  # Ensure efficient column access
    labels = np.asarray(labels)  # Ensure labels are a NumPy array

    for column_index in range(data.shape[1]):
        if sparse.issparse(data):
            column = data[:, column_index]
            if column.nnz == 0:  # Skip empty columns
                ig_values.append(0)
                continue
            column = column.toarray().ravel()
        else:
            column = data[:, column_index]
        ig = round(_information_gain(column, labels), 6)
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
    # Input are non-negative uint32 count vectors. Upcast to uint64 to avoid overflow
    # np.linalg.norm promotes to float internally, and so does final uint64 / (float * float).
    # NOTE: This function currently assumes integer input - if float scores were reintroduced, this section would need to be updated (branched on dtype).
    i = i.astype(np.uint64)
    j = j.astype(np.uint64)
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

    Notes
    -----
    For the gain ratio definition, see :func:`_gain_ratio`.
    """
    gr_values = []
    if sparse.issparse(data):
        data = data.tocsc()  # Ensure efficient column access
    labels = np.asarray(labels)  # Ensure labels are a NumPy array

    for column_index in range(data.shape[1]):
        if sparse.issparse(data):
            column = data[:, column_index]
            if column.nnz == 0:  # Skip empty columns
                gr_values.append(0)
                continue
            column = column.toarray().ravel()
        else:
            column = data[:, column_index]
        gr = _gain_ratio(column, labels)
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
