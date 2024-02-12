from abc import ABC

import networkx as nx
import numpy as np
from sklearn.metrics import classification_report
from sklearn.naive_bayes import BernoulliNB

from hfs.helpers import checkData, getRelevance
from hfs.metrics import conditional_mutual_information
from hfs.selectors import HierarchicalEstimator


class LazyHierarchicalFeatureSelector(HierarchicalEstimator, ABC):
    """
    Abstract class used for all lazy hierarchical feature selection methods.

    Every method should implement the method select_and_predict.
    """

    def __init__(self, hierarchy: np.ndarray = None):  # todo G = None
        """
        Initialize a LazyHierarchicalFeatureSelector with the required data.

        Parameters
        ----------
        hierarchy : np.ndarray
            The hierarchy graph as an adjacency matrix.
        """
        self.hierarchy = hierarchy

    def fit(self, X, y=None):
        """
        Implementing the fit function for Sklearn compatibility.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = self._validate_data(
            X,
        )
        if y is not None:
            if not isinstance(y, np.ndarray):
                try:
                    y = np.asarray(y)
                except Exception as e:
                    raise ValueError(f"Invalid input for y: {e}")
            self.n_classes_ = np.unique(y).shape[0]

        return self

    def fit_selector(self, X_train, y_train, X_test, columns=None):
        """
        Fit LazyHierarchicalFeatureSelector class.

        Due to laziness fitting of parameters as well
        as predictions are obtained per instance.

        Parameters
        ----------
        X_train : {numpy array} of shape (n_samples, n_features)
            The training input samples.
        X_test : {numpy array} of shape (n_samples, n_features)
            The test input samples.
            converted into a sparse ``csc_matrix``.
        y_train : array-like of shape (n_samples, n_levels)
            The target values, i.e., hierarchical class labels for classification.
        """
        # Create DAG
        self.n_features_in_ = X_train.shape[1]
        self.n_classes_ = np.unique(y_train).shape[0]

        self._set_hierarchy()
        self._hierarchy.remove_node("ROOT")
        if columns:
            self._columns = columns
        else:
            self._columns = list(range(self.n_features_in_))

        mapping = {value: index for index, value in enumerate(self._columns)}
        self._hierarchy = nx.relabel_nodes(self._hierarchy, mapping)

        self._xtrain = X_train
        self._ytrain = y_train
        self._xtest = X_test

        self._features = np.zeros(shape=X_test.shape)
        self._feature_length = np.zeros(self._xtest.shape[1], dtype=int)

        # Validate data
        checkData(self._hierarchy, self._xtrain, self._ytrain)

        # Get relevance of each node
        self._relevance = {}
        for node in self._hierarchy:
            self._relevance[node] = getRelevance(self._xtrain, self._ytrain, node)
        self._sorted_relevance = sorted(self._relevance, key=self._relevance.get)

        self._instance_status = {}
        for node in self._hierarchy:
            self._instance_status[node] = 1

    def select_and_predict(
        self, predict=True, saveFeatures=False, estimator=BernoulliNB()
    ):
        """
        Select features lazy for each test instance amd optionally predict target value of test instances.

        To be implemented by children.

        Parameters
        ----------
        predict :   {bool}
            true if predictions shall be obtained.
        saveFeatures : {bool}
            true if features selected for each test instance shall be saved.
        estimator : sklearn-compatible estimator.
            Estimator to use for predictions.

        Returns
        -------
        predictions for test input samples
            if predict = false, returns empty array
        """
        pass

    def _get_nonredundant_features(self, idx):
        """
        Get nonredundant features without relevance score.

        Basic functionality of the algorithm HIP proposed by Wan & Freitas.

        Parameters
        ----------
        idx : int
            Index of test instance for which the features shall be selected.
        """
        for node in self._hierarchy:
            self._instance_status[node] = 1
        for node in self._hierarchy:
            if self._xtest[idx][node] == 1:
                for anc in self._hierarchy.predecessors(node):
                    self._instance_status[anc] = 0
            else:
                for desc in self._hierarchy.successors(node):
                    self._instance_status[desc] = 0

    def _get_nonredundant_features_relevance(self, idx):
        """
        Get nonredundant features based on relevance score.

        Basic functionality of the HNB algorithm proposed by Wan & Freitas.

        Parameters
        ----------
        idx :
            Index of test instance for which the features shall be selected.
        """
        for node in self._hierarchy:
            self._instance_status[node] = 1
        for node in self._hierarchy:
            if node == "ROOT":
                continue
            if self._xtest[idx][node] == 1:
                for anc in nx.ancestors(self._hierarchy, node):
                    if self._relevance[anc] <= self._relevance[node]:
                        self._instance_status[anc] = 0
            else:
                for desc in nx.descendants(self._hierarchy, node):
                    if self._relevance[desc] <= self._relevance[node]:
                        self._instance_status[desc] = 0

    def _get_nonredundant_features_mr(self, idx):
        """
        Get nonredundant features based on the MR considering all pathes.

        Basic functionality of the HIP algorithm proposed by Wan & Freitas.

        Parameters
        ----------
        idx :
            Index of test instance for which the features shall be selected.
        """

        top_sort = list(nx.topological_sort(self._hierarchy))
        reverse_top_sort = reversed(top_sort)
        mr = {}

        for node in top_sort:
            # correctness: as each predecessor lies on the same path as the current node
            # one can be removed.
            #
            # not several nodes per path:
            # contradiction - nodes a and b (a<b) selected and are on the same path.
            # each node a+1..b-1 between would save a as most relevant node:
            # 0: mr[a] = a
            # 1: mr[a+1] = a+1, if rel[mr[pred]=a] > rel[mr[a+1]=a+1]: mr[a+1]=a
            # and status[a+1] = remove
            # i: mr[a+i] = a+i, if rel[mr[pred]=a] > rel[mr[a+i]=a+i]: mr[a+i]=a
            # and status[a+i] = remove
            # When b is processed, mr[b-1] = a, so either rel[a]>rel[b] -> mr[b]=b is removed
            # and set to a OR mr[b-1]=a is removed and mr[b] stays b.
            #
            # at least one node per path: As there is a maximum relevant node in each path
            # this node will stay mr[node] = node and not be exchanged through mr[pred].
            # Then the condition is never met on this path
            # so self._instance_status[mr[node]] = 0 never executed.
            #
            mr[node] = []
            more_rel_nodes = [node]
            if self._xtest[idx][node]:
                # preds are 1 because of 0-1-propagation
                for pred in self._hierarchy.predecessors(node):
                    # get most relevant nodes seen on the paths until current node
                    for _mr in mr[pred]:
                        # if their is a node on the path more important then current node
                        if self._relevance[_mr] > self._relevance[node]:
                            self._instance_status[node] = 0
                            # save this node for next iterations (steps on path)
                            more_rel_nodes.append(_mr)
                        else:
                            # save current node as most important.
                            # there can be several paths, in this case, several nodes are saved
                            self._instance_status[_mr] = 0
                            more_rel_nodes.append(node)
            mr[node] = more_rel_nodes

        for node in reverse_top_sort:
            mr[node] = []
            more_rel_nodes = [node]
            if not self._xtest[idx][node]:
                mr[node] = node
                for suc in self._hierarchy.successors(node):
                    # get most relevant nodes seen on paths until current node
                    for _mr in mr[suc]:
                        if self._relevance[_mr] > self._relevance[node]:
                            # each node not selected will removed
                            self._instance_status[node] = 0
                            more_rel_nodes.append(_mr)
                        else:
                            self._instance_status[_mr] = 0
                            more_rel_nodes.append(node)
            mr[node] = more_rel_nodes

    def _get_top_k(self):
        """
        Get k highest-ranked features by relevance.
        """
        counter = 0
        for node in reversed(self._sorted_relevance):
            if node == "ROOT":
                continue
            if (counter < self.k or not self.k) and self._instance_status[node]:
                counter += 1
            else:
                self._instance_status[node] = 0

    def _build_mst(self):
        """
        Build minium spanning tree for each possible edge in the feature tree.
        """
        edges = self._hierarchy.edges
        self._edge_status = np.zeros((self.n_features_in_, self.n_features_in_))
        self._cmi = np.zeros((self.n_features_in_, self.n_features_in_))
        self._sorted_edges = []
        for node1 in self._hierarchy.nodes:
            for node2 in self._hierarchy.nodes:
                if node1 == node2:
                    continue
                self._cmi[node1][node2] = conditional_mutual_information(
                    self._xtrain[:, node1], self._xtrain[:, node2], self._ytrain
                )
                self._edge_status[node1][node2] = 1
        sorted_indices = np.argsort(self._cmi, axis=None)
        for index in sorted_indices:
            coordinates = divmod(index, self.n_features_in_)

            if coordinates[0] < coordinates[1]:
                self._sorted_edges.append(coordinates)

    def _get_nonredundant_features_from_mst(self, idx):
        """
        Get nonredundant features from MST.

        Basic functionality of the algorithm TAN proposed by Wan & Freitas.

        Parameters
        ----------
        idx : int
            Index of test instance for which the features shall be selected.
        """
        UDAG = nx.Graph()

        for node1 in self._hierarchy:
            self._instance_status[node1] = 0
            for node2 in self._hierarchy:
                self._edge_status[node1][node2] = 1

        representants = [i for i in range(self.n_features_in_)]
        members = {}
        for i in range(self.n_features_in_):
            members[i] = [i]

        # get paths
        reachable_nodes = {}
        for node in self._hierarchy:
            reachable_nodes[node] = []
            for des in nx.descendants(self._hierarchy, node):
                reachable_nodes[node].append(des)
        # select edges
        for edge in self._sorted_edges:
            if (
                self._edge_status[edge[0]][edge[1]]
                # check redundancy: same path and same value
                and (
                    self._xtest[idx][edge[0]] != self._xtest[idx][edge[1]]
                    or (
                        edge[0] not in reachable_nodes[edge[1]]
                        and edge[1] not in reachable_nodes[edge[1]]
                    )
                )
                # check if circle in UDAG using the property, that edge (a,b) infers circle iff a und b
                # are members of the same component
                and representants[edge[0]] != representants[edge[1]]
            ):
                UDAG.add_edge(edge[0], edge[1])
                self._edge_status[edge[0]][edge[1]] = 0

                # merge: change the representatives of the smaller component
                if len(members[representants[edge[0]]]) <= len(
                    members[representants[edge[1]]]
                ):
                    for m in members[edge[0]]:
                        representants[m] = representants[edge[1]]
                        members[representants[edge[1]]].append(m)
                else:
                    for m in members[edge[1]]:
                        representants[m] = representants[edge[0]]
                        members[representants[edge[0]]].append(m)

                # remove all edges with redundant ancestors or descendants of e0 and e1
                for selected_node in [edge[0], edge[1]]:
                    for neighbor_node in nx.ancestors(self._hierarchy, selected_node):
                        if (
                            self._xtest[idx][selected_node]
                            == self._xtest[idx][neighbor_node]
                        ):
                            # alternative: collect all and then delete in sorted_edges
                            self._edge_status[:, neighbor_node] = 0
                            self._edge_status[neighbor_node][:] = 0
                    for neighbor_node in nx.descendants(self._hierarchy, selected_node):
                        if (
                            self._xtest[idx][selected_node]
                            == self._xtest[idx][neighbor_node]
                        ):
                            self._edge_status[:, neighbor_node] = 0
                            self._edge_status[neighbor_node][:] = 0

                self._instance_status[edge[0]] = 1
                self._instance_status[edge[1]] = 1

    def _predict(self, idx, estimator):
        """
        Predicts for an instance of the test set.

        Parameters
        ----------
        idx : int
            Index of test instance which shall be predicted.
        estimator : sklearn-compatible estimator
                    Estimator to use for predictions.

        Returns
        -------
        prediction : bool
            prediction of test instance's target value.
        """
        features = [nodes for nodes, status in self._instance_status.items() if status]
        features_in_dataset = []
        for feature in features:
            features_in_dataset.append(self._columns.index(feature))
        clf = estimator
        clf.fit(self._xtrain[:, features_in_dataset], self._ytrain)
        return clf.predict(self._xtest[idx][features_in_dataset].reshape(1, -1))

    def get_score(self, ytest, predictions):
        """
        Returns score of the predictions.

        Note that recall of the positive class is known as “sensitivity”;
        recall of the negative class is “specificity”

        Parameters
        ----------
        ytest : 1d array-like, or label indicator array / sparse matrix
            truth values of y.
        predictions : 1d array-like, or label indicator array / sparse matrix
            obtained predictions.

        Returns
        -------
        report : dict
            metrics of prediction.
        """
        avg_feature_length = 0
        for idx in range(0, self._xtest.shape[0] - 1):
            avg_feature_length += self._feature_length[idx] / self._xtrain.shape[1]
        avg_feature_length = avg_feature_length / (len(self._feature_length))
        score = classification_report(
            y_true=ytest, y_pred=predictions, output_dict=True
        )
        score["sensitivityxspecificity"] = float(score["0"]["recall"]) * float(
            score["1"]["recall"]
        )
        score["compression"] = avg_feature_length

        return score

    def get_features(self):
        """
        Get selected features.

        Parameters
        ----------
        None

        Returns
        -------
        features : numpy array
            Boolean value at index states if feature is selected.
        """
        return self._features
