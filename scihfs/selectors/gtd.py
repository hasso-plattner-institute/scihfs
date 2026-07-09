"""
Greedy Top Down Feature Selector.
"""

import networkx as nx
import numpy as np
import scipy.sparse as sp

from scihfs.metrics import gain_ratio, information_gain
from scihfs.selectors import EagerHierarchicalFeatureSelector


class GreedyTopDownSelector(EagerHierarchicalFeatureSelector):
    """Greedy Top Down feature selection method proposed by Lu et al. 2013.

    The features are selected choosing nodes from the hierarchy that
    score in the heuristic function and aren't an ancestor or descendant
    of a node with a higher score. The heuristic function ranking the
    nodes is a relevance metric, and GTD can use both gain ratio (``"GR"``)
    and information gain (``"IG"``) as metric.
    This feature selection method is intended for hierarchical data.
    Therefore, it inherits from the EagerHierarchicalFeatureSelector.
    """

    def __init__(
        self,
        hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None,
        iterate_first_level: bool = True,
        heuristic_function: str = "GR",
    ):
        """Initializes a GreedyTopDownSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph. See ``HierarchicalEstimator.__init__``
                    for the accepted formats.
        iterate_first_level : bool
                            The feature selection algorithm proposed by Lu et
                            al. assumes that the hierarchy has a tree
                            structure. If it is a DAG this parameter can be set
                            to False to achieve similiar behaviour than in the
                            original algorithm.
        heuristic_function : str
                            The relevance metric used as the heuristic function
                            to rank the nodes. GTD accepts both "GR" (gain
                            ratio) and "IG" (information gain). Default is "GR".
        """
        super().__init__(hierarchy)
        self.iterate_first_level = iterate_first_level  # TODO: warning for DAG
        self.heuristic_function = heuristic_function

    def _select(self, X, y):
        """The actual GTD feature selection algorithm."""
        if sp.issparse(X):
            X = X.tocsr()
        self.calculate_heuristic_function(X, y)

        # either start from ROOT or the nodes on the first level.
        if self.iterate_first_level:
            top_level_nodes = self._hierarchy_graph.successors("ROOT")
        else:
            top_level_nodes = ["ROOT"]

        for node in top_level_nodes:
            branch_nodes = list(nx.descendants(self._hierarchy_graph, node))
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
                remove_nodes = list(nx.descendants(self._hierarchy_graph, selected))
                ancestor_nodes = list(nx.ancestors(self._hierarchy_graph, selected))
                remove_nodes.extend(ancestor_nodes)
                if "ROOT" in remove_nodes:
                    remove_nodes.remove("ROOT")
                branch_nodes = [node for node in branch_nodes if node not in remove_nodes]

    def calculate_heuristic_function(self, X, y):
        if self.heuristic_function == "GR":
            relevance_values = gain_ratio(X, y)
        elif self.heuristic_function == "IG":
            relevance_values = information_gain(X, y)
        else:
            raise ValueError(
                f"Unknown heuristic_function {self.heuristic_function!r}; "
                'expected "GR" (gain ratio) or "IG" (information gain).'
            )
        self.heuristic_function_values_ = dict(zip(self._columns, relevance_values))
