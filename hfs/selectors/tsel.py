"""
TSEL Feature Selector.
"""
import numpy as np
from networkx.algorithms.dag import descendants
from sklearn.utils.validation import check_X_y

from hfs.helpers import get_paths
from hfs.metrics import lift
from hfs.selectors.eagerHierarchicalFeatureSelector import (
    EagerHierarchicalFeatureSelector,
)


class TSELSelector(EagerHierarchicalFeatureSelector):
    """A tree-based feature selection method for hierarchical features.

    This hierarchical feature selection methods was proposed by Jeong and
    Myaeng in 2013. The features are selected by choosing the most
    representative nodes from each path and filtering these nodes further
    by removing parents with children that were also selected.
    """

    def __init__(
        self, hierarchy: np.ndarray = None, use_original_implementation: bool = True
    ):
        """Initializes a TSELSelector.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix. The
                    feature selection method is intended for a hierarchy
                    graph that has a tree structure.
        use_original_implementation: bool
                    Should the original implementation from the
                    paper be used. If False, a slightly different
                    interpretation of the algorithm is used. Default
                    is True.
        """
        super().__init__(hierarchy)
        self.use_original_implementation = use_original_implementation

    def fit(self, X, y, columns=None):
        """Fitting function that sets self.representatives\_.

        The number of columns in X and the number of nodes in the hierarchy
        are expected to be the same and each column should be mapped to
        exactly one node in the hierarchy with the columns parameter.
        After fitting self.representatives\_ includes the names of all
        nodes from the hierarchy that are left after feature selection.
        The features are selected by choosing the most
        representative nodes from each path and filtering these nodes further
        by removing parents with children that were also selected.

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
        X, y = check_X_y(X, y, accept_sparse=True)

        super().fit(X, y, columns)

        # Feature Selection Algorithm
        paths = get_paths(self._hierarchy)
        lift_values = lift(X, y)
        self._node_to_lift = {
            column_name: lift_values[index]
            for index, column_name in enumerate(self._columns)
        }
        self.representatives_ = self._find_representatives(paths)

        self.is_fitted_ = True
        return self

    def _find_representatives(self, paths):
        """ "Finds a representative node for each path.

        This is the first stage of the feature selection algorithm.
        In this stage two different implementation can be used.
        This is determined by the self.use_original_implementation
        parameter.

        Parameters
        ----------
        paths : list
                The paths for which the representative nodes should
                be found. This is a list of lists of node names.

        Returns
        -------
        list : A list of node names. This are the features chosen
            by the feature selection algorithm.
        """
        representatives = set()
        for path in paths:
            path.remove("ROOT")
            max_node = (
                self._select_from_path1(path)
                if self.use_original_implementation
                else self._select_from_path2(path)
            )
            representatives.add(max_node)
        return self._filter_representatives(representatives)

    def _select_from_path1(self, path: list[str]):
        """Finds the prepresentative node for a path.

        This is the implementation used in paper by Jeong and Myaeng.

        Parameters
        ----------
        paths : list
                The paths for which the representative nodes should
                be found. This is a list of lists of node names.

        Returns
        -------
        node : int
                The node selected as the representative for the given
                path.
        """
        for index, node in enumerate(path):
            if index == len(path) - 1:
                return node
            elif self._node_to_lift[node] >= self._node_to_lift[path[index + 1]]:
                return node

    def _select_from_path2(self, path: list[str]):
        """Finds the prepresentative node for a path.

        This is a different interpretation of the algorithm form the
        paper by Jeong and Myaeng. If multiple nodes are the maximum
        the node closest to the root is returned

        Parameters
        ----------
        paths : list
                The paths for which the representative nodes should
                be found. This is a list of lists of node names.

        Returns
        -------
        node : int
                The node selected as the representative for the given
                path.
        """
        max_node = max(path, key=lambda x: self._node_to_lift[x])
        return max_node

    def _filter_representatives(self, representatives: list[str]):
        """Filters the representative nodes selected in the previous stage.

        Parameters
        ----------
        representatives : list
                The list of previously selected nodes.

        Returns
        -------
        representatives : list
                The list of filtered representatives.
        """
        updated_representatives = []
        for node in representatives:
            selected_decendents = [
                descendent
                for descendent in descendants(self._hierarchy, node)
                if descendent in representatives
            ]
            if not selected_decendents:
                updated_representatives.append(node)
        return updated_representatives
