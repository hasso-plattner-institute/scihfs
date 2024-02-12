"MR-select feature selection"

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class MR(LazyHierarchicalFeatureSelector):

    """
    Select non-redundant features with the highest relevance on each path following the algorithm proposed by Wan and Freitas
    """

    def __init__(self, hierarchy=None):
        super(MR, self).__init__(hierarchy)
        """Initializes a MR-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray
                    The hierarchy graph as an adjacency matrix.
        """

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.
        The features are selected such that for each path only the most relevant features are preserved
        following the relevance score defined in helpers.py.

        Parameters
        ----------
        predict :  bool
            true if predictions shall be obtained
        saveFeatures : bool
            true if features selected for each test instance shall be saved.
        estimator : sklearn-compatible estimator
            Estimator to use for predictions


        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features_mr(idx)
            if predict:
                predictions = np.append(predictions, self._predict(idx, estimator)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
            self._feature_length[idx] = len(
                [nodes for nodes, status in self._instance_status.items() if status]
            )
            for node in self._hierarchy:
                self._instance_status[node] = 1
        return predictions
