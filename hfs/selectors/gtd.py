"""
Greedy Top Down Feature Selector.
"""
import numpy as np
from networkx import ancestors, descendants
from scipy.sparse import issparse
from sklearn.utils.validation import check_X_y

from hfs.metrics import gain_ratio
from hfs.selectors import EagerHierarchicalFeatureSelector


class GreedyTopDownSelector(EagerHierarchicalFeatureSelector):
    """Greedy Top Down feature selection method proposed by Lu et al. 2013.

    The features are selected choosing nodes from the hierarchy that
    score in the heuristic function and aren't an ancestor or descendant
    of a node with a higher score.
    This feature selection method is intended for hierarchical data.
    Therefore, it inherits from the EagerHierarchicalFeatureSelector.
    """

    def __init__(self, hierarchy: np.ndarray = None, iterate_first_level: bool = True):
        """Initializes a GreedyTopDownSelector.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix.
        iterate_first_level : bool
                            The feature selection algorithm proposed by Lu et
                            al. assumes that the hierarchy has a tree
                            structure. If it is a DAG this parameter can be set
                            to False to achieve similiar behaviour than in the
                            original algorithm."""
        super().__init__(hierarchy)
        self.iterate_first_level = iterate_first_level  # TODO: warning for DAG

    def fit(self, X, y, columns=None):
        """Fitting function that sets self.representatives\_.

        The number of columns in X and the number of nodes in the hierarchy
        are expected to be the same and each column should be mapped to
        exactly one node in the hierarchy with the columns parameter.
        After fitting self.representatives\_ includes the names of all
        nodes from the hierarchy that are left after feature selection.
        The features are selected choosing nodes from the hierarchy that
        score in the heuristic function and aren't an ancestor or
        descendant of a node with a higher score.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        columns: list or None, length n_features
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints. If this parameter is None the columns in X and
            the corresponding nodes in the hierarchy are expected to be in the
            same order.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X, y = check_X_y(X, y, accept_sparse=True)
        if issparse(X):
            X = X.tocsr()
        super().fit(X, y, columns)

        # Feature Selection Algorithm
        self.calculate_heuristic_function(X, y)
        self._fit()

        self.is_fitted_ = True
        return self

    def calculate_heuristic_function(self, X, y):
        gr_values = gain_ratio(X, y)
        self.heuristic_function_values_ = dict(zip(self._columns, gr_values))

    def _fit(self):
        self.representatives_ = []

        # either start from ROOT or the nodes on the first level.
        if self.iterate_first_level:
            top_level_nodes = self._hierarchy.successors("ROOT")
        else:
            top_level_nodes = ["ROOT"]

        for node in top_level_nodes:
            branch_nodes = list(descendants(self._hierarchy, node))
            if node != "ROOT":
                branch_nodes.append(node)

            # sort nodes in branch accaoring to heuristic function
            branch_nodes.sort(
                reverse=True, key=lambda x: self.heuristic_function_values_[x]
            )

            # select nodes with highest heuristic function value and remove
            # all their descendants and ancestors
            while branch_nodes:
                selected = branch_nodes.pop(0)
                self.representatives_.append(selected)
                remove_nodes = list(descendants(self._hierarchy, selected))
                ancestor_nodes = list(ancestors(self._hierarchy, selected))
                remove_nodes.extend(ancestor_nodes)
                if "ROOT" in remove_nodes:
                    remove_nodes.remove("ROOT")
                branch_nodes = [
                    node for node in branch_nodes if node not in remove_nodes
                ]
