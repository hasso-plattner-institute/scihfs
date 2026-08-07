"TAN feature selection"

import networkx as nx
import numpy as np
import scipy.sparse as sp

from scihfs.metrics import conditional_mutual_information

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class TAN(LazyHierarchicalFeatureSelector):
    """Lazy TAN classifier (Wan & Freitas). TODO: Insert reference.

    Builds a minimum spanning tree over the feature graph from the training
    data (conditional mutual information), then selects, per test instance, the
    non-redundant most-relevant features from that tree.
    """

    def _fit(self, X, y):
        # TEMPORARY densification. Will be removed upon overhaul of this class.
        if sp.issparse(self._xtrain):
            self._xtrain = self._xtrain.toarray()
        self._build_mst()

    def _build_mst(self):
        """Build the minimum spanning tree edges for the feature tree (train-only).

        Computes the conditional mutual information of every ordered node pair
        and the resulting relevance-sorted edge list ``self._sorted_edges``,
        both derived purely from the training data.
        """
        self._cmi = np.zeros((self.n_features_in_, self.n_features_in_))
        self._sorted_edges = []
        for node1 in self._hierarchy_graph.nodes:
            for node2 in self._hierarchy_graph.nodes:
                if node1 == node2:
                    continue
                self._cmi[node1][node2] = conditional_mutual_information(
                    self._xtrain[:, node1], self._xtrain[:, node2], self._ytrain
                )
        # The resolution of tied CMI values is architecture-dependent
        # (difference between x86 and arm64 observed). A stable sort
        # yields deterministic behavior.
        sorted_indices = np.argsort(self._cmi, axis=None, kind="stable")
        for index in sorted_indices:
            coordinates = divmod(index, self.n_features_in_)
            if coordinates[0] < coordinates[1]:
                self._sorted_edges.append(coordinates)

    def _select_features_per_instance(self, x_row):
        """Select the non-redundant most-relevant features from the MST.

        Parameters
        ----------
        x_row : numpy array of shape (n_features,)
            One test instance.

        Returns
        -------
        instance_status : dict
            The node->0/1 selection mask.
        """
        n_features = self.n_features_in_
        instance_status = {node: 0 for node in self._hierarchy_graph}
        edge_status = np.ones((n_features, n_features))

        representants = [i for i in range(n_features)]
        members = {i: [i] for i in range(n_features)}

        # get paths
        reachable_nodes = {
            node: list(nx.descendants(self._hierarchy_graph, node))
            for node in self._hierarchy_graph
        }
        # select edges
        for edge in self._sorted_edges:
            if (
                edge_status[edge[0]][edge[1]]
                # check redundancy: same path and same value
                and (
                    x_row[edge[0]] != x_row[edge[1]]
                    or (
                        edge[0] not in reachable_nodes[edge[1]]
                        and edge[1] not in reachable_nodes[edge[1]]
                    )
                )
                # check if circle in UDAG using the property, that edge (a,b) infers circle iff a und b
                # are members of the same component
                and representants[edge[0]] != representants[edge[1]]
            ):
                edge_status[edge[0]][edge[1]] = 0

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
                    for neighbor_node in nx.ancestors(
                        self._hierarchy_graph, selected_node
                    ):
                        if x_row[selected_node] == x_row[neighbor_node]:
                            # alternative: collect all and then delete in sorted_edges
                            edge_status[:, neighbor_node] = 0
                            edge_status[neighbor_node][:] = 0
                    for neighbor_node in nx.descendants(
                        self._hierarchy_graph, selected_node
                    ):
                        if x_row[selected_node] == x_row[neighbor_node]:
                            edge_status[:, neighbor_node] = 0
                            edge_status[neighbor_node][:] = 0

                instance_status[edge[0]] = 1
                instance_status[edge[1]] = 1
        return instance_status
