"HNB-select feature selection"

import numpy as np
from sklearn.naive_bayes import BernoulliNB

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class TAN(LazyHierarchicalFeatureSelector):
    """
    Select non-redundant features following the algorithm proposed by Wan and Freitas.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HNBs-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray
            The hierarchy graph as an adjacency matrix.
        """
        super(TAN, self).__init__(hierarchy)

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.
        It builds a minimal spanning tree (MST), by first adding all possible edges,
        that meets certain conditions (to remove redundancy and selecting most relevant features) to an undirected graph (UDAG).
        Afterwards features are obtained from the tree and can be used for prediction.

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
        self._build_mst()
        for idx in range(len(self._xtest)):
            self._get_nonredundant_features_from_mst(idx)
            if predict:
                predictions = np.append(predictions, self._predict(idx, estimator)[0])
            if saveFeatures:
                self._features[idx] = np.array(list(self._instance_status.values()))
            self._feature_length[idx] = len(
                [nodes for nodes, status in self._instance_status.items() if status]
            )
        return predictions
