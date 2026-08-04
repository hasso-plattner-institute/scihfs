"""
Different metric functions.
"""

import numpy as np
from numpy.linalg import norm
from scipy import sparse
from sklearn.metrics import mutual_info_score, recall_score


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


# ---------------------------------------------------------------------------
# The two functions below replace the pyitlib module (removed
# to keep dependency list short and because of lack of maintenance).
# The following code has been LLM-generated; it mirrors the estimation
# approach of pyitlib's information_mutual_conditional and was verified
# to be numerically equivalent to it before the module's removal.
# Corresponding unit tests were added to test_metrics.py with the dit library
# as an oracle to verify the correctness of this implementation.

# Original implementation: pyitlib, Copyright (c) 2016 Peter Foster under MIT
# License, https://github.com/pafoster/pyitlib


def _joint_entropy(variables) -> float:
    """Joint Shannon entropy (in bits) of one or more discrete variables.

    Parameters
    ----------
    variables : sequence of array-like, each shape (n_samples,)
        Aligned realisations of the discrete random variables.

    Returns
    ----------
    float : The joint entropy in bits.
    """
    observations = np.stack(variables)
    _, counts = np.unique(observations, axis=1, return_counts=True)
    probabilities = counts / observations.shape[-1]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def conditional_mutual_information(node1, node2, y):
    """Calculates conditional mutual information for two features given the target.

    Def.: ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)``

    H(...) is the joint entropy (base 2). Estimated via maximum
    likelihood from the observed frequencies; all inputs must be fully
    observed (there is no missing-data placeholder handling).

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
    return (
        _joint_entropy((node1, y))
        + _joint_entropy((node2, y))
        - _joint_entropy((node1, node2, y))
        - _joint_entropy((y,))
    )


# ---------------------------------------------------------------------------


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


def sensitivity_specificity_product(y_true, y_pred):
    """Product of sensitivity and specificity for a binary {0, 1} target.

    Common metric in the lazy hierarchical classifiers literature; ``score`` on
    the (lazy) classifiers only gives plain accuracy.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        The ground-truth labels (0/1).
    y_pred : array-like of shape (n_samples,)
        The predicted labels (0/1).

    Returns
    ----------
    float : sensitivity * specificity.
    """
    sensitivity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    return float(sensitivity) * float(specificity)


def mean_selected_fraction(masks):
    """Mean fraction of features selected per instance ("compression").

    Parameters
    ----------
    masks : array-like of shape (n_samples, n_features), dtype bool
        The per-instance selection masks, as returned by a lazy selector's
        ``select`` method.

    Returns
    ----------
    float : The mean fraction of selected features per instance, in [0, 1].
    """
    masks = np.asarray(masks, dtype=bool)
    return float(masks.mean())


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
    # Since np.corrcoef has no sparse support, densify the two (n_samples,)
    # columns at their time of comparison (and not the full feature matrix).
    if sparse.issparse(i):
        i = i.toarray().ravel()
    if sparse.issparse(j):
        j = j.toarray().ravel()
    return np.corrcoef(i, j)[0, 1]
