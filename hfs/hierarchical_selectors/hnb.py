"HNB feature selection"

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HNB(LazyHierarchicalFeatureSelector):

    """
    Select the k non-redundant features with the highest relevance following the algorithm proposed by Wan and Freitas.
    """

    def __init__(self, hierarchy=None, k=0):
        """Initializes a HNB-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray
            The hierarchy graph as an adjacency matrix.
        k : int
            The numbers of features to select.
        """

        super(HNB, self).__init__(hierarchy)
        self.k = k

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.
        It selects the top-k-ranked features, such that redundancy along each path is removed,
        in descending order of their individual predictive power measured by their relevance defined in helpers.py.

        Parameters
        ----------
        predict : bool
            true if predictions shall be obtained.
        saveFeatures : bool
            true if features selected for each test instance shall be saved.
        estimator : sklearn-compatible estimator
            Estimator to use for predictions.


        Returns
        -------
        predictions for test input samples, if predict = false, returns empty array.
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features_relevance(idx)
            self._get_top_k()
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
