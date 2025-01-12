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
        """
        Fits the selector for the HieAODE algorithm variants by calculating essential probability
        terms and storing them in a structured format for quick access during the computation of
        posterior probabilities. This method initializes and populates a dictionary of Conditional
        Probability Tables (CPTs), which are used across all variants of the HieAODE algorithm.

        This method must be called before performing any posterior probability calculations with
        the HieAODE algorithm variants, as it prepares the necessary probability data structures.

        Parameters:
            X_train (array-like): Training feature data. The shape should be (n_samples, n_features).
            y_train (array-like): Training target vector. The shape should be (n_samples,).
            X_test (array-like): Test feature data. The shape should be (n_samples, n_features).
            columns (list of str, optional): Specifies which columns in the feature data to consider
                for computing the probability tables. If None, all columns are used. Defaults to None.

        Attributes updated:
            self.cpts (dict): Contains keys mapping to different Conditional Probability Tables:
                - 'prior': Stores the class prior probabilities for a feature x_i and the class y
                  P(y, x_i) with shape (x_i, y, value).
                - 'prob_feature_given_class_and_parent': Stores the conditional probabilities for
                  a feature x_j given a class y and its parent x_i P(x_j | y, x_i) with shape
                  (x_j (feature), x_i (parent), class, value).
                - 'prob_feature_given_class': Stores the probabilities for an ancestor x_k of the
                  feature x_i given the class y P(x_k | y) with shape
                  (x_k (feature), class, value).

        Notes:
            - The probabilities are initialized to -1, indicating uncomputed probabilities.
            - The method should be invoked after any changes to the training dataset or when recalculating
              the model with different parameters or data.

        Returns:
            None: This method updates the model's state but does not return any value.
        """
        super(HieAODEBase, self).fit_selector(X_train, y_train, X_test, columns)
        self.cpts = dict(
            prior=np.full(
                (self.n_features_in_, self.n_classes_, 2),
                -1,
                dtype=float,
            ),
            prob_feature_given_class_and_parent=np.full(
                (self.n_features_in_, self.n_features_in_, self.n_classes_, 2, 2),
                -1,
                dtype=float,
            ),
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

        Parameters:
            sample (array-like): The data sample for which the product computation is intended.
                                 This could be a single data instance (e.g., a row from a dataset).
            parent_idx (int): The index of the parent feature x_i.
            ancestors (list of int): A list of indices representing the ancestors for the parent x_i.

        Returns:
            float: The result of the product computation.
        """
        raise NotImplementedError

    def prior_term(self, sample, parent_idx):
        """
        Computes the prior term for a given feature in a sample based on its class prior probabilities.
            This method leverages the pre-computed class prior probabilities stored in the 'prior' field of
            the 'cpts' dictionary.

            Parameters:
                sample (array-like): An array or list containing feature values for a single data instance.
                                     This is expected to be a part of a larger dataset used in model training or testing.
                parent_idx (int): The index of the feature within `sample` for which the prior term is being calculated.
                                  This index corresponds to the position of the feature in the dataset's feature array.

            Returns:
                float: The computed prior term, which is the product of the class prior probabilities for the
                       given feature value at `parent_idx`.
        """
        self.calculate_class_prior(feature_idx=parent_idx)
        return np.prod(self.cpts["prior"][parent_idx, :, sample[parent_idx]])

    def ancestors_product(self, sample, ancestors, use_positive_only=False):
        """
        Computes the product of probabilities for ancestor features from a given sample. The probabilities
        are derived from pre-computed conditional probability tables for each ancestor feature.

        The product can be calculated using all ancestors or restricted to only those ancestors with a positive
        value, depending on the `use_positive_only` flag.

        Parameters:
            sample (array-like): An array containing feature values for a single data instance. This array
                                 should correspond to the feature set described by the `ancestors` parameter.
            ancestors (list of int): A list of indices representing the ancestor features for which the product
                                     of probabilities is to be computed.
            use_positive_only (bool, optional): If True, the product calculation includes only those ancestors
                                                whose value in the `sample` is positive (typically 1). Defaults to False.

        Returns:
            numpy.ndarray: An array of the product of probabilities for each class. If there are no ancestors,
                           or all ancestors are filtered out when `use_positive_only` is True, this method returns
                           an array of zeros for each class.
        """
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
        """
        Computes the product of conditional probabilities for descendant features of a specified parent
        feature in a given sample.

        The function first identifies all features that are considered descendants (i.e., not the parent
        or any of the ancestors specified). It then calculates the conditional probabilities
        P(x_j=sample[descendant_idx] | y, x_i=sample[parent_idx]) for each descendant feature based on the
        current sample values.

        Parameters:
            sample (array-like): An array containing feature values for a single data instance.
            parent_idx (int): The index of the parent feature.
            ancestors (list of int): A list of indices representing the ancestor features.
            use_positive_only (bool, optional): If True, only descendants with a positive value (typically 1)
                                                in the sample are included in the product calculation. Defaults to False.

        Returns:
            numpy.ndarray: An array representing the product of conditional probabilities for each class. If there are
                           no eligible descendants (or all are filtered out when `use_positive_only` is True),
                           this method returns an array of zeros for each class.
        """
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
        """
        Computes the product of probabilities for negative values of descendant features in relation to a specified
            parent feature in a sample. This method focuses on cases where descendant feature values are zero.

            The function identifies all features that are not the specified parent or listed as ancestors as
            potential descendants. It calculates the product of their probabilities given they have a negative
            value (typically represented as 0).

            Parameters:
                sample (array-like): An array containing feature values for a single data instance. This array
                                     should match the feature set described by the model.
                parent_idx (int): The index of the parent feature within `sample` against which descendants are
                                  evaluated.
                ancestors (list of int): Indices representing the ancestor features which are not considered as
                                         descendants.

            Returns:
                numpy.ndarray: An array of the product of probabilities for each class considering only negative
                               (zero) values of descendant features. If no descendants are found or none have negative
                               values, this function returns an array of zeros for each class.
        """
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
        """
        Calculates and updates the class prior probabilities for a specific feature based on the training dataset.
        This method directly modifies the 'prior' field in the 'cpts' dictionary for a given feature index. The
        probabilities are computed for each class and each possible binary value of the feature.

        The probabilities are calculated as the ratio of the number of samples in each class having a specific
        feature value to the total number of samples. This calculation is performed only if the probability has
        not already been computed and stored (-1 indicates uncomputed).

        Parameters:
            feature_idx (int): The index of the feature for which to compute class priors. This index refers to
                               the column in the training feature array `_xtrain`.
        """
        n_samples = self._ytrain.shape[0]
        for c in range(self.n_classes_):
            for value in range(2):
                if self.cpts["prior"][feature_idx][c][value] == -1:
                    value_sum = np.sum(
                        (self._ytrain == c) & (self._xtrain[:, feature_idx] == value)
                    )
                    self.cpts["prior"][feature_idx][c][value] = value_sum / n_samples

    def calculate_prob_feature_given_class(self, feature):
        """
            Calculates and updates the conditional probabilities P(x_k | y) for a specific feature x_k across all classes y.
        This method adjusts the 'prob_feature_given_class' field in the 'cpts' dictionary for the specified feature,
        using a Laplace smoothing technique to ensure no zero probabilities.

        The probabilities are computed for each class and each possible value of the feature (assumed binary).
        Laplace smoothing helps manage cases where a feature value might not appear with a class in the training data.

        Parameters:
            feature (int): The index of the feature in the dataset for which to compute the conditional probabilities.
                           This index corresponds to the column in the training feature array `_xtrain`.
        """
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
        """
            Calculates and updates the conditional probabilities P(x_j | y, x_i = parent_value) for a given
        feature x_j and its parent x_i across all classes y. This method modifies the
        'prob_feature_given_class_and_parent' field in the 'cpts' dictionary for the specified feature
        index and parent index combination.

        Probabilities are computed incorporating Laplace smoothing to adjust for cases where certain combinations
        of feature values and classes may not be represented in the training data.

        Parameters:
            feature_idx (int): Index of the feature (child) for which to compute the conditional probabilities.
            parent_idx (int): Index of the parent feature. Conditional probabilities are computed given this parent feature.
        """
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
