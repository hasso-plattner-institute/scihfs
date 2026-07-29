"MR-select feature selection"

import networkx as nx

from .lazyHierarchicalFeatureSelector import LazyHierarchicalFeatureSelector


class MR(LazyHierarchicalFeatureSelector):
    """MR (Most Relevant): Lazy classifier from Wan et al., 2015, Algorithm 2.

    Selects, for each path, the most relevant non-redundant features following
    the per-node relevance score.
    """

    def _fit(self, X, y):
        self._compute_relevance(X, y)

    def _select_features_per_instance(self, x_row):
        """Select, per path, the most relevant non-redundant features.

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
        top_sort = list(nx.topological_sort(self._hierarchy_graph))
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
            # so instance_status[mr[node]] = 0 never executed.
            #
            mr[node] = []
            more_rel_nodes = [node]
            if x_row[node]:
                # preds are 1 because of 0-1-propagation
                for pred in self._hierarchy_graph.predecessors(node):
                    # get most relevant nodes seen on the paths until current node
                    for _mr in mr[pred]:
                        # if there is a node on the path more important than current node
                        if self._relevance[_mr] > self._relevance[node]:
                            instance_status[node] = 0
                            # save this node for next iterations (steps on path)
                            more_rel_nodes.append(_mr)
                        else:
                            # save current node as most important.
                            # there can be several paths, in this case, several nodes are saved
                            instance_status[_mr] = 0
                            more_rel_nodes.append(node)
            mr[node] = more_rel_nodes

        for node in reverse_top_sort:
            mr[node] = []
            more_rel_nodes = [node]
            if not x_row[node]:
                for suc in self._hierarchy_graph.successors(node):
                    # get most relevant nodes seen on paths until current node
                    for _mr in mr[suc]:
                        if self._relevance[_mr] > self._relevance[node]:
                            # each node not selected will be removed
                            instance_status[node] = 0
                            more_rel_nodes.append(_mr)
                        else:
                            instance_status[_mr] = 0
                            more_rel_nodes.append(node)
            mr[node] = more_rel_nodes
        return instance_status
