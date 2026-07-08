"""
TSEL Feature Selector.
"""

import numpy as np
from networkx.algorithms.dag import descendants

from scihfs.helpers import get_paths
from scihfs.metrics import lift
from scihfs.selectors.eagerHierarchicalFeatureSelector import (
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
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph, given either as a dense adjacency
                    matrix (``np.ndarray``), a sparse adjacency matrix
                    (``scipy.sparse``), or as directly as digraph (``networkx.DiGraph``, with optional node names that can match the columns in X).
                    The feature selection method is intended for a
                    hierarchy graph that has a tree structure.
        use_original_implementation: bool
                    Should the original implementation from the
                    paper be used. If False, a slightly different
                    interpretation of the algorithm is used. Default
                    is True.
        """
        super().__init__(hierarchy)
        self.use_original_implementation = use_original_implementation

    def _select(self, X, y):
        """The actual TSEL feature selection algorithm."""
        paths = get_paths(self._hierarchy_graph)
        lift_values = lift(X, y)
        self._node_to_lift = {
            column_name: lift_values[index]
            for index, column_name in enumerate(self._columns)
        }
        self.representatives_ = self._find_representatives(paths)

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
                for descendent in descendants(self._hierarchy_graph, node)
                if descendent in representatives
            ]
            if not selected_decendents:
                updated_representatives.append(node)
        return updated_representatives
