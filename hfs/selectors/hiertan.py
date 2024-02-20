"HNB-select feature selection"

import random
import statistics

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.naive_bayes import BernoulliNB
from sklearn.utils.validation import check_X_y

from ..helpers import compute_aggregated_values, get_leaves, get_paths
from ..metrics import (
    conditional_mutual_information,
    information_gain,
    pearson_correlation,
)
from .eagerHierarchicalFeatureSelector import EagerHierarchicalFeatureSelector


class HierTan(EagerHierarchicalFeatureSelector):
    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas (2022).
    """

    def __init__(
        self,
        hierarchy: np.ndarray = None,
        seed: int = None
        # relevance_metric: str = "IG",
        # similarity_threshold=0.99,
        # use_hfe_extension=False,
        # preprocess_numerical_data=False,
    ):
        """Initializes a SHSELSelector.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix.
        # relevance_metric : str
        #             The relevance metric to use in the initial selection
        #             stage of the algorithm. The options ore "IG" for
        #             information gain and "Correlation". Default is IG.
        # similarity_threshold : float
        #             The similarity threshold to use in the initial selection
        #             stage of the algorithm. This can be a number between
        #             0 an 1. Default is 0.99.
        # use_hfe_extension : bool
        #             If True the HFE algorithm proposed by Oudah and Henschel is
        #             used. Set relevance_metric to "Correlation" when using this
        #             extension. Default is False.
        # preprocess_numerical_data : False
        #             If True the data is preprocessed by adding up the child values.
        #             This method is used in the HFE extension algorithm which
        #             expects numerical data. If binary data is used it is
        #             recommended to set this parameter to False. Default is False.

        """
        super().__init__(hierarchy)
        random.seed(seed)


    def fit(self, X, y, columns=None):
        """Fitting function that sets self.representatives\_.

        The number of columns in X and the number of nodes in the hierarchy
        are expected to be the same and each column should be mapped to
        exactly one node in the hierarchy with the columns parameter.
        After fitting self.representatives\_ includes the names of all
        nodes from the hierarchy that are left after feature selection.
        The features are selected by removing features with
        parents that have a similar relevance and removing features with
        lower than average information gain for each path from leaf to
        root.

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
        if sparse.issparse(X):
            X = X.tocsr()
        super().fit(X, y, columns)

        # Feature Selection Algorithm
        # self._calculate_relevance(X, y)
        self._fit(X, y)

        self.is_fitted_ = True
        return self

    def _fit(self, X, y):
        """The feature selection algorithm."""
        self._build_hdc_spanning_tree()
        self._train_tan(X, y)

    # TODO: use correct name, check input
    def classify(self, X):
        # P(x | parent(x), y)
        # x_conditional_probs[x,0,0,1] is probability that x == 0, conditional on parent(x) == 0 and y == 1
        probs = np.tile(self._prob_y, (X.shape[0],2))
        for feature_idx in range(len(self._hierarchy.nodes)):
            # if self._feature_tree.nodes[feature_idx] != self._root:
            parent_idx = self._feature_tree.nodes[feature_idx].ancestors[0]
            feature_factor = np.zeros(2)
            # use values of numpy array as index lookup?
            feature_factor += X[:,feature_idx] * X[:,parent_idx] * self._conditional_probs_x[feature_idx, 1, 1]
            feature_factor += (1 - X[:,feature_idx]) * X[:,parent_idx] * self._conditional_probs_x[feature_idx, 0, 1]
            feature_factor += X[:,feature_idx] * (1 - X[:,parent_idx]) * self._conditional_probs_x[feature_idx, 1, 0]
            feature_factor += (1 - X[:,feature_idx]) * (1 - X[:,parent_idx]) * self._conditional_probs_x[feature_idx, 0, 0]
            probs *= feature_factor
        return np.argmax(probs, axis=1)
                

    def _build_hdc_spanning_tree(self):
        """
        Build hierarchical dependency constrained spanning tree from the complete feature graph based on Wan & Freitas 2022 (Algorithm 2).
        Contrary to their naming, this is NOT a maximum spanning tree, even with regard to their additional constraints.
        """
        nodes = list(self._hierarchy.nodes)
        get_cmi = lambda node1, node2: conditional_mutual_information(self._xtrain[:, node1], self._xtrain[:, node2], self._ytrain)
        weighted_edges = [(node1, node2, {'w': get_cmi(node1,node2)}) for i, node1 in enumerate(nodes) for node2 in nodes[:i]]
        self._cmi_graph = nx.Graph()
        self._cmi_graph.nodes = self._hierarchy.nodes
        self._cmi_graph.add_edges_from(weighted_edges)
        sorted_edges = sorted(self._cmi_graph.edges(data=True), key=lambda edge: edge[2].get('w'), reverse=True)

        UF = nx.utils.UnionFind(nodes)
        DTREE = nx.DiGraph()
        UDEG = nx.Graph()
        DTREE.nodes = UDEG.nodes = nodes
        for node1, node2, _ in sorted_edges:
            # check if DTREE remains cycle-free
            if UF[node1] is not UF[node2]:
                n_from = n_to = None
                # check if hierarchy imposes direction
                if nx.has_path(self._hierarchy, node1, node2):
                    if DTREE.in_degree(node2) == 0:
                        n_from = node1, n_to = node2
                elif nx.has_path(self._hierarchy, node2, node1):
                    if DTREE.in_degree(node1) == 0:
                        n_from = node2, n_to = node1
                # else, check if DTREE imposes direction
                else:
                    # direction not imposed
                    if DTREE.in_degree(node2) == 0 and DTREE.in_degree(node1) == 0:
                        UDEG.add_edge(node1, node2)
                        UF.union(node1, node2)
                        continue
                    # direction imposed
                    elif DTREE.in_degree(node2) == 0:
                        n_from = node1, n_to = node2
                    elif DTREE.in_degree(node1) == 0:
                        n_from = node2, n_to = node1
                    # else: edge cannot be added

                # add edge if possible
                if n_from is not None:
                    UF.union(n_from, n_to)
                    DTREE.add_edge(n_from, n_to)
                    # propagation routine
                    propagate_edges = UDEG.edges(n_to)
                    for edge in propagate_edges:
                        if DTREE.in_degree(edge[0]):
                            n_from = edge[0], n_to = edge[1]
                        else:
                            n_from = edge[1], n_to = edge[0]
                        DTREE.add_edge(n_from, n_to)
                        UDEG.remove_edge(n_from, n_to)
                        propagate_edges.extend(UDEG.edges(n_to))
                    # check if already tree
                    if UF.weights[node1] == len(nodes):
                        break
        # insert undecided edges with random direction
        for edge in UDEG.edges:
            edge = random.sample(edge, 2)
            DTREE.add_edge(edge[0], edge[1])

        self._feature_tree = DTREE

    def _train_tan(self, X, y):
        OUTPUT_CLASSES = 2 # fix to binary classification
        FEATURE_CLASSES = 2 # fix to binary features
        self._feature_tree
        self._cmi_graph
        root = [v for v in self._feature_tree.nodes if v.in_degree == 0][0]
        y_abs = np.bincount(y, minlength=OUTPUT_CLASSES)
        self._prob_y = y_abs / np.sum(y_abs)
        # P(x | parent(x), y)
        # x_conditional_probs[x,0,0,1] is probability that x == 0 conditional on parent(x) == 0 and y == 1
        self._conditional_probs_x = np.empty((X.shape[1], FEATURE_CLASSES, FEATURE_CLASSES, OUTPUT_CLASSES), dtype=np.float64)
        
        for y_val in range(OUTPUT_CLASSES):
            relevant_y = y == y_val
            for x_conditional_val in range(FEATURE_CLASSES):
                candidates = (X == x_conditional_val).T & relevant_y
                candidate_count = np.sum(candidates, axis=1)
                for x_wanted_val in range(FEATURE_CLASSES):
                    wanted = X.T == x_wanted_val
                    for feature_idx in range(len(self._hierarchy.nodes)):
                        if self._feature_tree.nodes[feature_idx] == root:
                            # handle special case
                            self._conditional_probs_x[feature_idx, x_wanted_val, x_conditional_val, y_val] \
                                = np.sum(wanted[feature_idx]) / np.sum(relevant_y)
                        else:
                            parent = self._feature_tree.nodes[feature_idx].ancestors[0]
                            self._conditional_probs_x[feature_idx, x_wanted_val, x_conditional_val, y_val] \
                                = np.sum(wanted[feature_idx] & candidates[parent]) / candidate_count[parent]

    