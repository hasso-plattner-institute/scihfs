"""
Hill Climbing Feature Selectors.
"""

import math
from abc import abstractmethod

import networkx as nx
import numpy as np
import scipy.sparse as sp

# Numerical input is not supported and previous related code commented out throughout this file, such as the 'dataset_type' variable with "binary" and "numerical" branches.
# Restore the `normalize_score` import here when reintroducing.
from scihfs.helpers import compute_aggregated_values, get_leaves
from scihfs.metrics import cosine_similarity
from scihfs.selectors import EagerHierarchicalFeatureSelector


class HillClimbingSelector(EagerHierarchicalFeatureSelector):
    """Base class for hill climbing feature selection methods proposed by Wang.

    The feature selection methods are intended for hierarchical data.
    Therefore, this class inherits from the EagerHierarchicalFeatureSelector.
    """

    def __init__(
        self,
        hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None,
        alpha: float = 0.99,
        # dataset_type: str = "binary",
    ):
        """Initializes a HillClimbingSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph. See ``HierarchicalEstimator.__init__``
                    for the accepted formats.
        alpha: float
                A hyperparameter needed for the hill climbing methods.
                The default value is 0.99.

        Notes
        -----
        Numerical input is currently not supported. Corresponding code is retained (but inactive) for future reintroduction.
        This also applies to the `dataset_type` parameter.

        dataset_type: string, either "binary" or "numerical"
                A value indicating if the input dataset contains binary or
                numerical data. Default is "binary".
        """
        super().__init__(hierarchy)
        self.alpha = alpha
        # self.dataset_type = dataset_type

    def _select(self, X, y):
        """The feature selection algorithm framework which delegates
        the actual selection to ``_hill_climb``."""
        if sp.issparse(X):
            X = X.tocsr()
        self.y_ = y
        self._num_rows = X.shape[0]
        self.representatives_ = self._hill_climb(X)

    @abstractmethod
    def _hill_climb(self, X):
        """Performs the feature selection.

        This method needs to be implemented by the subclasses.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        """

    def _calculate_scores(self, X):
        """Calculate scores for each value in X.

        To calculate the scores the values X are summed up with
        the feature's children's values. If the dataset is of the
        type "numerical" the sums are normalized.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        score_matrix : numpy.ndarray, shape (n_samples, n_features)
                    The scores calculated for each value in X.
        """
        score_matrix = compute_aggregated_values(
            X.copy(), self._hierarchy_graph, self._columns
        )

        # if self.dataset_type == "numerical":
        #     normalized_matrix = np.zeros_like(score_matrix, dtype=float)
        #     for row_index in range(self._num_rows):
        #         for column_index in range(self.n_features_in_):
        #             score = score_matrix[row_index, column_index]
        #             normalized_matrix[row_index, column_index] = normalize_score(
        #                 score, max(score_matrix[row_index, :])
        #             )
        #     score_matrix = normalized_matrix
        return score_matrix

    @abstractmethod
    def _compare(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        """Compare two samples from the dataset.

        This method needs to be implemented by the subclasses.
        It should return a score that can be used for comparison.

        Parameters
        ----------
        sample_i : int
            A row index for a sample from the dataset.
        sample_j : int
            A row index for another sample from the dataset.
        feature_set: list
                    A list of nodes that are in the feature set that is
                    currently being evaluated.
        """

    def _comparison_matrix(self, feature_set: list[int]):
        """Creates matrix to compare the individual samples from the dataset.

        Parameters
        ----------
        feature_set : list
                A list of nodes that are in the feature set that is
                currently being evaluated.

        Returns
        -------
        distances : numpy.ndarray, shape (num_rows, num_rows)
                    An matrix with distances or similarities to compare all
                    rows from the dataset with eachother.
        """
        distances = np.zeros((self._num_rows, self._num_rows), dtype=float)
        for row_i in range(self._num_rows):
            for row_j in range(self._num_rows):
                if distances[row_i, row_j] == 0:
                    distance = self._compare(row_i, row_j, feature_set)
                    distances[row_i, row_j] = distance
                    distances[row_j, row_i] = distance
        return distances

    @abstractmethod
    def _fitness_function(self, comparison_matrix: np.ndarray) -> float:
        """Evaluates a comparison matrix into a single fitness score.

        This method needs to be implemented by the subclasses.
        """


class TopDownSelector(HillClimbingSelector):
    """Hill climbing top down feature selection method.

    This feature selection method was proposed by Wang et al. in 2002.
    The features are selected by going through the feature graph from
    top to bottom, replacing parent nodes with their children and
    evaluating the resulting feature set with a fitness function.
    The method is intended for hierarchical data. Therefore, it inherits
    from the EagerHierarchicalFeatureSelector.
    """

    def _hill_climb(self, X) -> list[int]:
        """Performs the feature selection.

        The features are selected by going through the feature graph from
        top to bottom, replacing parent nodes with their children and
        evaluating the resulting feature set with a fitness function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        optimal_feature_set : list
                    A list of nodes (int or str) that represent the
                    optimal features set.
        """
        self._score_matrix = self._calculate_scores(X)

        # Start with nodes on first level after virtual root node
        optimal_feature_set = set(self._hierarchy_graph.successors("ROOT"))
        fitness = 0
        best_fitness = 0
        best_feature_set = None

        while True:
            for node in optimal_feature_set:
                children = list(self._hierarchy_graph.successors(node))
                if children:
                    # Replace the current node with its children and
                    # evaluate the resulting feature set using the
                    # fitness function.
                    temporary_feature_set = optimal_feature_set.copy()
                    temporary_feature_set.remove(node)
                    temporary_feature_set.update(children)
                    distances = self._comparison_matrix(temporary_feature_set)
                    temporary_fitness = self._fitness_function(distances)
                    if (temporary_fitness) > best_fitness:
                        best_fitness = temporary_fitness
                        best_feature_set = temporary_feature_set

            if best_fitness > fitness:
                optimal_feature_set = best_feature_set
                fitness = best_fitness
            else:
                break
        return list(optimal_feature_set)

    def _compare(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        return self._calculate_distance(sample_i, sample_j, feature_set)

    def _calculate_distance(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        distance = 0
        for column in feature_set:
            column_index = self._column_index(column)
            # To accomodate large int (squares+sum), the uint32 input (_score_matrix coming from compute_aggregated_values) is transiently transformed to unbounded integers.
            # The sqrt at the end transforms the int input to float output.
            difference = int(self._score_matrix[sample_i, column_index]) - int(
                self._score_matrix[sample_j, column_index]
            )
            distance += difference * difference
        return math.sqrt(distance)

    def _fitness_function(self, comparison_matrix: np.ndarray) -> float:
        result = 0
        row_indices = range(self._num_rows)
        for row_index in row_indices:
            same_class = [
                sample for sample in row_indices if self.y_[sample] == self.y_[row_index]
            ]
            other_class = [x for x in row_indices if x not in same_class]

            nominator = sum([comparison_matrix[x, row_index] for x in other_class])
            denominator = 1 + self.alpha * sum(
                [comparison_matrix[sample, row_index] for sample in same_class]
            )
            result += nominator / denominator

        return result


class BottomUpSelector(HillClimbingSelector):
    """Hill climbing bottom up feature selection method.

    This feature selection method was proposed by Wang et al. in 2003.
    The features are selected by going through the feature graph from
    bottom to top, replacing child nodes with their parent node and
    evaluating the resulting feature set with a fitness function.
    The method is intended for hierarchical data. Therefore, it inherits
    from the EagerHierarchicalFeatureSelector.
    """

    def __init__(
        self,
        hierarchy: np.ndarray | sp.sparray | sp.spmatrix | nx.DiGraph | None = None,
        alpha: float = 0.01,
        k: int = 5,
        # dataset_type: str = "binary",
    ):
        """Initializes a BottomUpSelector.

        Parameters
        ----------
        hierarchy : np.ndarray, scipy.sparse array/matrix or nx.DiGraph
                    The hierarchy graph. See ``HierarchicalEstimator.__init__``
                    for the accepted formats.
                    For this feature selection method to work as
                    intended the graph needs to be a tree.
        alpha: float
                A hyperparameter needed for the feature selection. In
                the paper by Wang et al. this parameter is called beta.
                The default value is 0.01.
        k : int
                A hyperparameter needed to determine the k nearest
                neighbors during the feature selection algorithm.
                The default value is 5.

        Notes
        -----
        Numerical input is currently not supported. Corresponding code is retained (but inactive) for future reintroduction.
        This also applies to the `dataset_type` parameter.

        dataset_type: string, either "binary" or "numerical"
                A value indicating if the input dataset contains binary or
                numerical data. Default is "binary".
        """
        # dataset_type disabled:
        super().__init__(hierarchy, alpha=alpha)
        self.k = k

    def _hill_climb(self, X) -> list[int]:
        """Performs the feature selection.

        The features are selected by going through the feature graph from
        bottom to top, replacing child nodes with their parent node and
        evaluating the resulting feature set with a fitness function.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        optimal_feature_set : list
                    A list of nodes (int or str) that represent the
                    optimal features set.
        """
        # The kNN implementation in this file requires n_samples > k_neighbors.
        if self._num_rows <= self.k:
            raise ValueError(
                f"BottomUpSelector needs n (samples) > k (neighbors) for the k-nearest-"
                f"neighbors step, but got n_samples={self._num_rows} with "
                f"k={self.k}."
            )
        self._score_matrix = self._calculate_scores(X)

        # Start with the leaves.
        current_feature_set = get_leaves(self._hierarchy_graph)
        if current_feature_set == ["ROOT"] or current_feature_set == []:
            return []
        current_fitness = self._fitness_function(
            self._comparison_matrix(current_feature_set)
        )

        unvisited = set(current_feature_set)

        while unvisited:
            temporary_feature_set = current_feature_set.copy()
            node = unvisited.pop()
            parent = list(self._hierarchy_graph.predecessors(node))[
                0
            ]  # This does not work with a DAG.
            if parent != "ROOT":
                # Replace the current node and its siblings with their
                # parent node.
                temporary_feature_set.append(parent)
                children = list(self._hierarchy_graph.successors(parent))
                updated_feature_set = [
                    node for node in temporary_feature_set if node not in children
                ]
                temporary_fitness = self._fitness_function(
                    self._comparison_matrix(updated_feature_set)
                )
                if temporary_fitness < current_fitness:
                    current_feature_set = temporary_feature_set
                    current_fitness = temporary_fitness
                    unvisited = set(current_feature_set)

        return current_feature_set

    def _compare(
        self,
        sample_i: int,
        sample_j: int,
        feature_set: list[int],
    ):
        return self._calculate_similarity(sample_i, sample_j, feature_set)

    def _calculate_similarity(self, sample_i: int, sample_j: int, feature_set: list[int]):
        if "ROOT" in feature_set:
            feature_set = feature_set.remove("ROOT")
        row_i = self._score_matrix[sample_i, feature_set]
        row_j = self._score_matrix[sample_j, feature_set]
        return cosine_similarity(row_i.flatten(), row_j.flatten())

    def _fitness_function(self, comparison_matrix: np.ndarray) -> float:
        # number_of_leaf_nodes is the alpha value from paper.
        number_of_leaf_nodes = len(get_leaves(self._hierarchy_graph))
        if number_of_leaf_nodes == 0:
            number_of_leaf_nodes = 1

        threshold_index = self._num_rows - self.k - 1
        k_nearest_neigbors = [
            list(
                np.argpartition(comparison_matrix[row, :], threshold_index)[
                    threshold_index:
                ]
            )
            for row in range(self._num_rows)
        ]

        count = 0
        for row in range(self._num_rows):
            for neighbor in k_nearest_neigbors[row]:
                if self.y_[neighbor] == self.y_[row] and neighbor != row:
                    count += 1

        result = count * (
            1
            + self.alpha
            * (number_of_leaf_nodes - self.n_features_in_)
            / number_of_leaf_nodes
        )
        return result
