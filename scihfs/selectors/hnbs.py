"HNB-select feature selection"

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HNBs(LazyHierarchicalFeatureSelector):
    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HNBs-Selector.

        Parameters
        ----------
        hierarchy: np.ndarray
            The hierarchy graph as an adjacency matrix.
        """
        super(HNBs, self).__init__(hierarchy)

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazily for each test instance and optionally predict the target value of test instances.
        It selects the features such that redundancy along each path is removed.

        Parameters
        ----------
        predict : bool, default=True
            Whether predictions should be obtained.
        saveFeatures : bool, default=False
            Whether features selected for each test instance should be saved.
        estimator : sklearn-compatible estimator, default=BernoulliNB()
            The estimator to use for predictions.

        Returns
        -------
        np.ndarray
            Predictions for test input samples. If `predict` is False, returns an empty array.
        """
        predictions = np.array([])
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features_relevance(idx)
            if predict:
                predictions = np.append(predictions, self._predict(idx, estimator)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
            self._feature_length[idx] = len(
                [nodes for nodes, status in self._instance_status.items() if status]
            )
        return predictions
