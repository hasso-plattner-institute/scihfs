from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class HIP(LazyHierarchicalFeatureSelector):
    """HIP (Hierarchical Information-Preserving): Lazy classifier from Wan et al., 2015, Algorithm 1.

    Essentially a compression which keeps for each path only the most detailed
    positive or the most abstract negative feature -- for the subset of
    per-instance features.
    """

    def _select_features_per_instance(self, x_row):
        """Keep, per path, the deepest positive or highest negative feature.

        Parameters
        ----------
        x_row : numpy array of shape (n_features,)
            One test instance.

        Returns
        -------
        instance_status : dict
            The node->0/1 selection mask.
        """
        instance_status = {node: 1 for node in self._hierarchy_graph}
        for node in self._hierarchy_graph:
            if x_row[node] == 1:
                for anc in self._hierarchy_graph.predecessors(node):
                    instance_status[anc] = 0
            else:
                for desc in self._hierarchy_graph.successors(node):
                    instance_status[desc] = 0
        return instance_status
