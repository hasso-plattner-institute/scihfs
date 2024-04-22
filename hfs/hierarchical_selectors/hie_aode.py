import networkx as nx
import numpy as np
from sklearn.naive_bayes import BernoulliNB
import abc

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector

SMOOTHING_FACTOR = 1
PRIOR_PROBABILITY = 0.5


class HieAODEBase(LazyHierarchicalFeatureSelector, abc.ABC):
    """
    Base Class for selecting non-redundant features following the algorithms proposed by Wan and Freitas.
    """

    def __init__(self, hierarchy=None):
        """Initializes a HieAODE-Selector.

        Parameters
        ----------
        hierarchy : np.ndarray
            The hierarchy graph as an adjacency matrix.
        """
        self.cpts = dict()
        super(HieAODEBase, self).__init__(hierarchy)

    def fit_selector(self, X_train, y_train, X_test, columns=None):
        super(HieAODEBase, self).fit_selector(X_train, y_train, X_test, columns)
        self.cpts = dict(
            # P(y, x_i)
            # Shape (x_i, y, value)
            prior=np.full(
                (self.n_features_in_, self.n_classes_, 2),
                -1,
                dtype=float,
            ),

            # P(x_j|y, x_i)
            # Shape (x_j (feature), x_i (parent), class, value)
            prob_feature_given_class_and_parent=np.full(
                (self.n_features_in_, self.n_features_in_, self.n_classes_, 2, 2),
                -1,
                dtype=float,
            ),
            # P(x_k|y)
            # Shape (x_k (feature), class, value)
            prob_feature_given_class=np.full(
                (self.n_features_in_, self.n_classes_, 2), -1, dtype=float
            ),
        )

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance and optionally predict target value of test instances
        using the one of de HieAODE algorithms by Wan and Freitas

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
        n_samples = self._xtest.shape[0]
        sample_sum = np.zeros((n_samples, self.n_classes_))
        for sample_idx in range(n_samples):
            sample = self._xtest[sample_idx]
            for parent_idx in range(self.n_features_in_):
                ancestors = list(nx.ancestors(self._hierarchy, parent_idx))
                feature_product = self.compute_product(sample, parent_idx, ancestors)
                sample_sum[sample_idx] = np.add(sample_sum[sample_idx], feature_product)

        y = np.argmax(sample_sum, axis=1)
        return y if predict else np.array([])

    @abc.abstractmethod
    def compute_product(self, sample, parent_idx, ancestors):
        """
        Subclasses should implement this method to define how feature products are computed.
        """
        raise NotImplementedError

    def prior_term(self, sample, parent_idx):
        self.calculate_class_prior(feature_idx=parent_idx)
        return np.prod(self.cpts["prior"][parent_idx, :, sample[parent_idx]])

    def ancestors_product(self, sample, ancestors, use_positive_only=False):
        # Calculate probabilities for each ancestor
        for ancestor_idx in ancestors:
            self.calculate_prob_feature_given_class(feature=ancestor_idx)
        # Handle case with no ancestors
        if len(ancestors) <= 0:
            return np.zeros(self.n_classes_)
        # Extract values for ancestors from the sample
        ancestors_value = sample[ancestors]
        # Extract corresponding CPT entries for the specific ancestors
        # and their values
        ancestors_cpt = self.cpts["prob_feature_given_class"][
            ancestors, :, ancestors_value
        ]
        # If using only positive ancestors, filter the CPTs accordingly
        if use_positive_only and np.any(ancestors_value == 1):
            ancestors_cpt = ancestors_cpt[ancestors_value == 1]

        return np.prod(ancestors_cpt, axis=0)

    def descendants_product(
        self, sample, parent_idx, ancestors, use_positive_only=False
    ):
        descendants = [
            feature
            for feature in range(self.n_features_in_)
            if feature != parent_idx and feature not in ancestors
        ]

        for descendant_idx in descendants:
            # Calculates P(x_j=sample[descendant_idx]|y, x_i=sample[parent_idx])
            self.calculate_prob_feature_given_class_and_parent(
                feature_idx=descendant_idx, parent_idx=parent_idx
            )
        if len(descendants) <= 0:
            return np.zeros(self.n_classes_)

        descendants_value = sample[descendants]
        feature_value = sample[parent_idx]
        descendants_cpt = self.cpts["prob_feature_given_class_and_parent"][
            descendants,
            parent_idx,
            :,
            feature_value,
            descendants_value,
        ]

        if use_positive_only and np.any(descendants_value == 1):
            descendants_cpt = descendants_cpt[descendants_value == 1]

        return np.prod(
            descendants_cpt,
            axis=0,
        )

    def descendants_product_negative(self, sample, parent_idx, ancestors):
        descendants = [
            feature
            for feature in range(self.n_features_in_)
            if feature != parent_idx and feature not in ancestors
        ]

        for descendant_idx in descendants:
            self.calculate_prob_feature_given_class(feature=descendant_idx)

        if len(descendants) <= 0:
            return np.zeros(self.n_classes_)

        descendants_value = sample[descendants]

        descendants_cpt = self.cpts["prob_feature_given_class"][
            descendants, :, descendants_value
        ]

        if np.any(descendants_value == 0):
            descendants_cpt = descendants_cpt[descendants_value == 0]
            return np.prod(descendants_cpt, axis=0)
        else:
            return np.zeros(self.n_classes_)

    def calculate_class_prior(self, feature_idx):
        n_samples = self._ytrain.shape[0]
        for c in range(self.n_classes_):
            for value in range(2):
                if self.cpts["prior"][feature_idx][c][value] == -1:
                    value_sum = np.sum(
                        (self._ytrain == c) & (self._xtrain[:, feature_idx] == value)
                    )
                    self.cpts["prior"][feature_idx][c][value] = value_sum / n_samples

    def calculate_prob_feature_given_class(self, feature):
        # Calculate P(x_k | y) where x_k=ascendant and y = c
        for c in range(self.n_classes_):
            p_class = np.sum(self._ytrain == c)
            for value in range(2):
                p_class_ascendant = np.sum(
                    (self._ytrain == c) & (self._xtrain[:, feature] == value)
                )
                self.cpts["prob_feature_given_class"][feature][c][value] = (
                    p_class_ascendant + SMOOTHING_FACTOR * PRIOR_PROBABILITY
                ) / (p_class + SMOOTHING_FACTOR)

    def calculate_prob_feature_given_class_and_parent(self, feature_idx, parent_idx):
        for c in range(self.n_classes_):
            for parent_value in range(2):
                # Calculate P(y, x_i = parent_value)
                mask = (self._xtrain[:, parent_idx] == parent_value) & (
                    self._ytrain == c
                )
                p_class_feature = np.sum(mask)
                for feature_value in range(2):
                    if feature_idx != parent_idx:
                        # Calculate P(y, x_i = parent_value, x_j = feature_value)
                        descendant = self._xtrain[:, feature_idx]
                        p_class_feature_descendant = np.sum(
                            descendant[mask] == feature_value
                        )
                        prob_descendant_given_c_feature = (
                            p_class_feature_descendant
                            + SMOOTHING_FACTOR * PRIOR_PROBABILITY
                        ) / (p_class_feature + SMOOTHING_FACTOR)

                        self.cpts["prob_feature_given_class_and_parent"][feature_idx][
                            parent_idx
                        ][c][feature_value][
                            parent_value
                        ] = prob_descendant_given_c_feature


class HieAODE(HieAODEBase):
    def compute_product(self, sample, parent_idx, ancestors):
        feature_product = np.multiply(
            self.ancestors_product(sample=sample, ancestors=ancestors),
            self.descendants_product(
                sample=sample, parent_idx=parent_idx, ancestors=ancestors
            ),
        )
        feature_product = np.multiply(
            feature_product,
            self.prior_term(sample=sample, parent_idx=parent_idx),
        )

        return feature_product


class HieAODEPlus(HieAODEBase):
    def compute_product(self, sample, parent_idx, ancestors):
        product = np.multiply(
            self.descendants_product(
                sample=sample,
                parent_idx=parent_idx,
                ancestors=ancestors,
                use_positive_only=True,
            ),
            self.descendants_product_negative(
                sample=sample,
                parent_idx=parent_idx,
                ancestors=ancestors,
            ),
        )
        product = np.multiply(
            product,
            self.ancestors_product(
                sample=sample, ancestors=ancestors, use_positive_only=True
            ),
        )
        product = np.multiply(
            product, self.prior_term(sample=sample, parent_idx=parent_idx)
        )
        return product


class HieAODEPlusPlus(HieAODEBase):
    def compute_product(self, sample, parent_idx, ancestors):
        feature_product = np.multiply(
            self.ancestors_product(
                sample=sample, ancestors=ancestors, use_positive_only=True
            ),
            self.descendants_product(
                sample=sample,
                parent_idx=parent_idx,
                ancestors=ancestors,
                use_positive_only=True,
            ),
        )
        feature_product = np.multiply(
            feature_product,
            self.prior_term(sample=sample, parent_idx=parent_idx),
        )
        return feature_product


class HieAODELite(HieAODEBase):
    def compute_product(self, sample, parent_idx, ancestors):
        feature_product = np.multiply(
            self.descendants_product(
                sample=sample, parent_idx=parent_idx, ancestors=ancestors
            ),
            self.prior_term(sample=sample, parent_idx=parent_idx),
        )
        return feature_product


class HieAODELitePlus(HieAODEBase):
    def compute_product(self, sample, parent_idx, ancestors):
        feature_product = np.multiply(
            self.descendants_product(
                sample=sample,
                parent_idx=parent_idx,
                ancestors=ancestors,
                use_positive_only=True,
            ),
            self.descendants_product_negative(
                sample=sample, parent_idx=parent_idx, ancestors=ancestors
            ),
        )
        feature_product = np.multiply(
            feature_product,
            self.prior_term(sample=sample, parent_idx=parent_idx),
        )
        return feature_product


class HieAODELitePlusPlus(HieAODEBase):
    def compute_product(self, sample, parent_idx, ancestors):
        feature_product = np.multiply(
            self.descendants_product(
                sample=sample,
                parent_idx=parent_idx,
                ancestors=ancestors,
                use_positive_only=True,
            ),
            self.prior_term(sample=sample, parent_idx=parent_idx),
        )
        return feature_product
