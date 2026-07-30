"""Base class for lazy hierarchical feature selection classifiers."""

from abc import ABC, abstractmethod

import networkx as nx
import numpy as np
import scipy.sparse as sp
from scipy.special import logsumexp
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils.validation import check_is_fitted, validate_data

from scihfs.helpers import check_binary_target, check_bool_dtype, get_relevance
from scihfs.selectors.base import HierarchyMixin


class _MaskedBernoulliNB(BernoulliNB):
    """A ``BernoulliNB`` that can score an instance on a feature subset.

    Naive Bayes class for binary classification identical to sklearn's own, with the
    only variation that it can score a single instance on a (masked) subset of the
    features, and that it can store the precomputed "feature is 0" log-probabilities
    for that scoring (instead of recomputing them on every instance).
    """

    def fit(self, X, y):
        super().fit(X, y)
        # Precompute the "feature is 0" log-probabilities.
        self._neg_log_prob = np.log(1 - np.exp(self.feature_log_prob_))
        return self

    def predict_proba_masked(self, x_row, columns):
        """Class probabilities for ``x_row`` (a single instance from the test set)
        using only the selected ``columns``.

        Calculates the joint log-likelihood of the fitted BernoulliNB on the selected
        columns, then normalises it to yield probabilities over the two classes.
        Required for predict_proba AND predict.

        Parameters
        ----------
        x_row : numpy array of shape (n_features,), bool
            One test instance.
        columns : list of int
            The selected feature columns. An empty list means "no evidence" and yields
            the training majority class.

        Returns
        -------
        proba : numpy array of shape (n_classes,)
            The normalised class probabilities, ordered by ``classes_`` (0,1).
        """
        x = x_row.astype(float)[columns]
        pos = self.feature_log_prob_[:, columns]
        neg = self._neg_log_prob[:, columns]
        jll = self.class_log_prior_ + (x * pos + (1 - x) * neg).sum(axis=1)
        return np.exp(jll - logsumexp(jll))


class LazyHierarchicalFeatureSelector(
    ClassifierMixin, HierarchyMixin, BaseEstimator, ABC
):
    """Abstract base class for lazy hierarchical feature-selection classifiers.

    Lazy means that the classifiers work on a per-(test-)instance basis, like kNN.
    Feature selection is embedded in these classification algorithms.
    The API is congruent with scikit-learn's:
    ``fit`` does the precomputation and all those steps that only need to be done once
    per training set, while ``predict`` does the actual per-instance (selection and)
    prediction.

    Lazy selectors are arrays-only: unlike the ``HierarchicalPreprocessor`` and
    the eager selectors they neither auto-derive the column->node mapping from
    DataFrame feature names nor support the transformer output API. Pass plain
    arrays (with an explicit ``columns`` mapping when column and node order
    differ); DataFrame-aware lazy input is left to a future redesign of the lazy
    interface.
    """

    def __init__(
        self,
        hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None,
    ):
        """Initializes a LazyHierarchicalFeatureSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
            The hierarchy graph. See ``HierarchicalEstimator.__init__`` for the
            accepted formats.
        """
        super().__init__(hierarchy)

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.input_tags.sparse = True
        return tags

    def fit(self, X, y, columns=None):
        """Fit the lazy classifier on the training data.

        Builds the hierarchy, fits a Bernoulli naive Bayes on all training
        columns and stores the training data; any computation of metrics or
        structures that is necessary for the classification of all test instances
        will need to be implemented in the subclasses' ``_fit`` (which is
        called here).

        Parameters
        ----------
        X : array-like or scipy.sparse (CSR) of shape (n_samples, n_features)
            The training input samples. Must be bool-dtype. Sparse input is
            densified internally (for now).
        y : array-like of shape (n_samples,)
            The target values.
        columns : list or None
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints; ``None`` assumes positional 1:1 ordering.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = validate_data(self, X, y, accept_sparse="csr")
        check_binary_target(y)
        if sp.issparse(X):
            X = X.toarray()
        check_bool_dtype(X)
        self.classes_ = np.unique(y)
        self.n_classes_ = self.classes_.shape[0]

        self._fit_hierarchy(columns)
        self._hierarchy_graph.remove_node("ROOT")
        node_to_column = {
            position: column for column, position in enumerate(self._columns)
        }
        self._hierarchy_graph = nx.relabel_nodes(self._hierarchy_graph, node_to_column)

        self._xtrain = X
        self._ytrain = y
        # CAUTION: Calling the BernoulliNB could be shifted to the subclasses' ``_fit``
        # in the future to allow replacement with other classifiers (if necessary).
        self._nb = _MaskedBernoulliNB(binarize=None).fit(X, y)

        self._fit(X, y)
        self.is_fitted_ = True
        return self

    def _fit(self, X, y):
        """Train-time hook for subclasses (default: no-op)."""

    def _compute_relevance(self, X, y):
        """Precompute the per-node relevance from the training data.

        Sets ``self._relevance`` (node -> relevance score). Called from the
        ``_fit`` hook of only those selectors (subclasses) that rank features by
        relevance. Those that don't, skip it.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples.
        y : numpy array of shape (n_samples,)
            The target values.
        """
        self._relevance = {
            node: get_relevance(X, y, node) for node in self._hierarchy_graph
        }

    def _sort_relevance(self):
        """Sort the hierarchy nodes by ascending relevance.

        Sets ``self._sorted_relevance``: The nodes from ``self._relevance`` ordered
        by their values.
        Requires ``_compute_relevance`` to have run first.
        """
        self._sorted_relevance = sorted(self._relevance, key=self._relevance.get)

    def _check_and_validate(self, X):
        """Shared function for input validation.

        Sparse input (CSR) is accepted and densified: the per-instance selection
        algorithms index single rows as dense, so the classifiers accept sparse
        at the boundary but work on a dense copy internally.

        Returns
        -------
        X : numpy array
            The validated (and densified) test input.
        """
        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse="csr", reset=False)
        if sp.issparse(X):
            X = X.toarray()
        check_bool_dtype(X)
        return X

    def _select_and_predict_proba(self, X, return_masks):
        """One per-instance sweep of selection + prediction.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The validated test input samples.
        return_masks : bool
            Whether to also build the per-instance selection masks.

        Returns
        -------
        proba : numpy array of shape (n_samples, n_classes)
            The per-instance class probabilities, columns ordered by ``classes_``.
        masks : numpy array of shape (n_samples, n_features), dtype bool, or None
            The per-instance selection masks when ``return_masks`` is set, else
            ``None``.
        """
        proba = np.empty((X.shape[0], self.n_classes_))
        masks = (
            np.zeros((X.shape[0], self.n_features_in_), dtype=bool)
            if return_masks
            else None
        )
        for idx in range(X.shape[0]):
            status = self._select_features_per_instance(X[idx])
            proba[idx] = self._predict_proba_per_instance(X[idx], status)
            if return_masks:
                for node, selected in status.items():
                    # Nodes are data-column indices after fit's relabel.
                    if selected:
                        masks[idx, node] = True
        return proba, masks

    def predict(self, X, return_masks=False):
        """Predict the target value for each instance in X.

        Argmaxes over the class probabilities per test instance.

        In order to avoid writing the selected features to ``self`` on a per-instance basis,
        this function allows the return of the selection masks ("the selected features") as
        a dedicated variable when ``return_masks`` is set to True (default False).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test input samples. Must be bool-dtype.
        return_masks : bool, default=False
            If True, also return the per-instance selection masks, built in the
            same sweep as the predictions (so both come from one pass instead of a
            separate ``select`` call). The masks are identical to ``select(X)``.

        Returns
        -------
        predictions : numpy array of shape (n_samples,)
            The predicted target values.
        masks : numpy array of shape (n_samples, n_features), dtype bool
            Returned only when ``return_masks`` is True; ``masks[i, j]`` is True
            iff feature column ``j`` was selected for test instance ``i``.
        """
        X = self._check_and_validate(X)
        proba, masks = self._select_and_predict_proba(X, return_masks=return_masks)
        predictions = self.classes_[np.argmax(proba, axis=1)]
        if return_masks:
            return predictions, masks
        return predictions

    def predict_proba(self, X):
        """Class probabilities for each (test) instance in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test input samples. Must be bool-dtype.

        Returns
        -------
        proba : numpy array of shape (n_samples, n_classes)
            The class probabilities, columns ordered by ``classes_`` (0,1).
        """
        X = self._check_and_validate(X)
        proba, _ = self._select_and_predict_proba(X, return_masks=False)
        return proba

    def select(self, X):
        """Return the per-instance selected-feature masks for X.

        The introspective counterpart to ``predict``: it runs the same
        per-instance selection but returns the boolean masks instead of
        predictions, without scoring the naive Bayes. Equivalent to the masks
        from ``predict(X, return_masks=True)``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The test input samples. Must be bool-dtype.

        Returns
        -------
        masks : numpy array of shape (n_samples, n_features), dtype bool
            ``masks[i, j]`` is True iff feature column ``j`` was selected for
            test instance ``i``.
        """
        X = self._check_and_validate(X)
        masks = np.zeros((X.shape[0], self.n_features_in_), dtype=bool)
        for idx in range(X.shape[0]):
            for node, selected in self._select_features_per_instance(X[idx]).items():
                # Nodes are data-column indices after fit's relabel.
                if selected:
                    masks[idx, node] = True
        return masks

    @abstractmethod
    def _select_features_per_instance(self, x_row):
        """Select the features for a single test instance.

        Parameters
        ----------
        x_row : numpy array of shape (n_features,)
            One test instance.

        Returns
        -------
        instance_status : dict
            Maps each hierarchy node to 1 (selected) or 0 (not selected).
        """

    def _predict_proba_per_instance(self, x_row, instance_status):
        """Class probabilities for one instance.

        Scores ``x_row`` against the fitted masked ``BernoulliNB`` on only the
        selected columns. After ``fit``'s relabel the hierarchy nodes are already
        data-column indices, so the selected nodes index the classifier directly.
        An empty selection scores on the class prior alone, i.e. yields the
        training class priors (whose argmax is the majority class).

        Parameters
        ----------
        x_row : numpy array of shape (n_features,)
            The test instance to classify.
        instance_status : dict
            The node->0/1 selection mask for this instance.

        Returns
        -------
        proba : numpy array of shape (n_classes,)
            The normalised class probabilities, ordered by ``classes_`` (0,1).
        """
        columns = [node for node, selected in instance_status.items() if selected]
        return self._nb.predict_proba_masked(x_row, columns)

    def _get_nonredundant_features_relevance(self, x_row):
        """Get nonredundant features based on relevance score.

        Basic functionality of HNB, but also required in other classifiers.

        Parameters
        ----------
        x_row : numpy array of shape (n_features,)
            One test instance.

        Returns
        -------
        instance_status : dict
            The node->0/1 selection mask.
        """
        instance_status = {node: 1 for node in self._hierarchy_graph}
        for node in self._hierarchy_graph:
            if x_row[node] == 1:
                for anc in nx.ancestors(self._hierarchy_graph, node):
                    if self._relevance[anc] <= self._relevance[node]:
                        instance_status[anc] = 0
            else:
                for desc in nx.descendants(self._hierarchy_graph, node):
                    if self._relevance[desc] <= self._relevance[node]:
                        instance_status[desc] = 0
        return instance_status

    def _get_top_k(self, instance_status):
        """Keep only the k highest-ranked of the selected features (ranked by relevance).

        Parameters
        ----------
        instance_status : dict
            The node->0/1 selection mask to prune in place.

        Returns
        -------
        instance_status : dict
            The pruned mask.
        """
        counter = 0
        for node in reversed(self._sorted_relevance):
            if (counter < self.k or not self.k) and instance_status[node]:
                counter += 1
            else:
                instance_status[node] = 0
        return instance_status
