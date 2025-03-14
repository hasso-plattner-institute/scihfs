"""
Collection of helper methods for the feature selection algorithms.
"""

import math
import warnings
from fractions import Fraction

import networkx as nx
import numpy as np
from networkx.algorithms.simple_paths import all_simple_paths


def get_relevance(xdata, ydata, node):
    """
    Gather relevance for a given node.

    Parameters
    ----------
    node : int
        Node for which the relevance should be obtained.
    xdata : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    ydata : array-like, shape (n_samples,)
            The target values. An array of int.
    """
    p1 = (
        Fraction(
            xdata[(xdata[:, node] == 1) & (ydata == 1)].shape[0],
            xdata[(xdata[:, node] == 1)].shape[0],
        )
        if xdata[(xdata[:, node] == 1)].shape[0] != 0
        else 0
    )
    p2 = (
        Fraction(
            xdata[(xdata[:, node] == 0) & (ydata == 1)].shape[0],
            xdata[(xdata[:, node] == 0)].shape[0],
        )
        if xdata[(xdata[:, node] == 0)].shape[0] != 0
        else 0
    )
    p3 = 1 - p1
    p4 = 1 - p2

    rel = (p1 - p2) ** 2 + (p3 - p4) ** 2
    return rel


def check_data(dag, x_data, y_data):
    """Checks whether the given dataset satisfies the 0-1-propagation on the DAG.

    The 0-1-propagation property states that if there is a directed edge (u, v)
    in the DAG, then whenever node u has a value of 1 in the dataset, node v
    must have a value of 1 for the same instance.

    Parameters
    ----------
    dag : networkx.DiGraph
        The Directed Acyclic Graph representing the hierarchy structure.
    x_data : numpy.ndarray
            An array containing the input features of the dataset.
    y_data : numpy.ndarray
            An array containing the corresponding output labels of the dataset.

    Raises
    ----------
    ValueError: If the dataset violates the 0-1-propagation property
    on any of the edges in the DAG.

    """
    data = np.column_stack((x_data, y_data))
    edges = list(nx.edge_dfs(dag, source=0, orientation="original"))
    for edge in edges:
        for idx in range(len(data)):
            if data[idx, edge[0]] == 0 and data[idx, edge[1]] == 1:
                raise ValueError(
                    f"Test instance {idx} violates 0-1-propagation \
                    on edge ({edge[0]}, {edge[1]})"
                    f"{data[idx]}"
                )


def get_leaves(graph: nx.DiGraph):
    """Get the leaf nodes from the given directed acyclic graph (DAG).

    A leaf node is a node in the graph that meets the following criteria:
    - It has no outgoing edges (out_degree == 0).
    - It has at least one incoming edge (in_degree > 0), indicating it
      has one or more parent nodes.

    Parameters
    ----------
    graph : networkx.DiGraph
            The Directed Acyclic Graph (DAG) from which the leaf nodes
            will be identified.

    Returns
    ----------
    leaves : list
            A list of leaf nodes found in the DAG.
    """
    leaves = [
        node
        for node in graph
        if graph.in_degree(node) > 0 and graph.out_degree(node) == 0
    ]
    return leaves


def shrink_dag(node_identifiers: list, digraph: nx.DiGraph):
    """Remove irrelevant leaf nodes from the given DAG.

    Parameters
    ----------
    node_identifiers : list
            A list containing node identifiers that are considered relevant
    digraph : networkx.DiGraph
            The Directed Acyclic Graph (DAG) from which irrelevant leaf nodes
            will be removed.

    Returns
    ----------
    digraph : networkx.DiGraph
            The resulting DAG after removing all irrelevant leaf nodes.
    """
    to_remove = {
        node
        for node in digraph.nodes()
        if _is_irrelevant_leaf(node, node_identifiers, digraph)
    }

    while to_remove:
        # Recompute the list of nodes to check (predecessors of removed nodes)
        to_check = {
            predecessor
            for node in to_remove
            for predecessor in digraph.predecessors(node)
        }
        digraph.remove_nodes_from(to_remove)
        to_remove = {
            node
            for node in to_check
            if _is_irrelevant_leaf(node, node_identifiers, digraph)
        }
    return digraph


def _is_irrelevant_leaf(node, node_identifiers, digraph):
    """
    Determine if a node is an irrelevant leaf in a directed acyclic graph (DAG).

    A node is considered an irrelevant leaf if:
    - It has no outgoing edges (i.e., it is a leaf node).
    - It is not included in the specified list of relevant node identifiers.
    - It is not the "ROOT" node

    Parameters
    ----------
    node : Any
        The node to evaluate.
    node_identifiers : list
        A list of node identifiers that are considered relevant and should not be removed.
    digraph : networkx.DiGraph
        The directed acyclic graph (DAG) being analyzed.

    Returns
    ----------
    bool
        True if the node is an irrelevant leaf; otherwise, False.
    """
    return (
        digraph.out_degree(node) == 0 and node != "ROOT" and node not in node_identifiers
    )


def connect_dag(node_identifiers: list, hierarchy: nx.DiGraph):
    """
    Connects digraph (DAG), so that every node not in node_identifiers is removed from the DAG, and an new edge with its predecessor is built.

    Parameters
    ----------
    node_identifiers: list
                A list of node identifiers that are considered relevant and should not be removed.
    hierarchy : networkx.DiGraph
                The Directed Acyclic Graph (DAG) representing the hierarchy.

    """
    top_sort = nx.topological_sort(hierarchy)

    # node i = 0: source is either in or not in, as there are no predecessors,
    # there should not be any artificial edge
    # i: for each pred there is a direct edge to the pred and iff pred not in x_ide
    #       also to their pred2. (it does not matter if pred2 is really in x, if it is not,
    #       the edge will be removed later anyway)
    # i+1: if i is in -> no artificial edge on this path needed
    #       if i is not -> artifical edge to every pred of i, so each path going through i
    #       will be continued, if i is removed later

    for node in list(top_sort):
        predecessors = list(hierarchy.predecessors(node))
        for predecessor in predecessors:
            new_connections = []
            if predecessor not in node_identifiers:
                for pred_of_pred in hierarchy.predecessors(predecessor):
                    new_connections.append(pred_of_pred)
                for new_connection in new_connections:
                    hierarchy.add_edge(new_connection, node)

    # remove all nodes (and edges) that are not in node_identifier
    nodes_to_remove = [
        node for node in hierarchy.nodes if node not in set(node_identifiers)
    ]
    hierarchy.remove_nodes_from(nodes_to_remove)
    return hierarchy


def add_virtual_root_node(hierarchy: nx.DiGraph):
    """Create a virtual root node to connect disjoint hierarchies.

    Parameters
    ----------
    hierarchy : networkx.DiGraph
                The Directed Acyclic Graph (DAG) representing the hierarchy.

    Returns
    ----------
    hierarchy : networkx.DiGraph
                The final hierarchy graph.
    """

    roots = [x for x in hierarchy.nodes() if hierarchy.in_degree(x) == 0]
    # create parent node to join hierarchies
    hierarchy.add_node("ROOT")
    if len(roots) > 1:
        warnings.warn(
            f"Hierarchy consists of multiple ({len(roots)}) disjoint hierarchies. "
        )
    for root_node in roots:
        hierarchy.add_edge("ROOT", root_node)
    return hierarchy


def get_paths(graph: nx.DiGraph, reverse=False):
    """Get all the paths from the "ROOT" node to the leaf nodes in the input graph.

    Parameters
    ----------
    graph : networkx.DiGraph
            The Directed Acyclic Graph (DAG) for which paths need to be found.
    reverse : bool
            If True, the order of nodes in each path will be reversed,
            effectively giving the paths from leaf nodes to the "ROOT" node.

    Returns
    ----------
    paths : list
            A list node lists which represent paths.
    """
    leaves = get_leaves(graph)
    paths = list(all_simple_paths(graph, "ROOT", leaves))
    if reverse:
        for path in paths:
            path.reverse()
    return paths


def get_columns_for_numpy_hierarchy(hierarchy: nx.DiGraph, num_columns: int):
    """Get mapping from hierarchy nodes to columns after hierarchy transformation.

    If each node in the hierarchy is named after a column's index this methods
    will give you the mapping from column index to node name of the node after
    the graph was transformed to a numpy array and back. During this
    transformation the node names are lost and afterwards each node is named
    after its index in hierarchy.nodes.

    Parameters
    ----------
    hierarchy : networkx.DiGraph
            The Directed Acyclic Graph (DAG) representing the hierarchy.
    num_columns : bool
            The number of columns in the dataset.

    Returns
    ----------
    columns : list
            A mapping from nodes to columns.
    """
    columns = []
    for node in range(num_columns):
        index = list(hierarchy.nodes()).index(node) if node in hierarchy.nodes else -1
        columns.append(index)
    return columns


def normalize_score(score, max_value):
    """Normalize the given score using logarithmic scaling and a maximum value.

    Parameters
    ----------
    score : float or int
            The score to be normalized.
    max_value : float or int
            The maximum of the scores in the corresponding row.

    Returns
    ----------
    float or int : The normalized score after applying logarithmic scaling.
    """
    if score != 0:
        score = math.log(1 + (score / max_value)) + 1
    return score


def compute_aggregated_values(X, hierarchy: nx.DiGraph, columns: list[int], node="ROOT"):
    """Recursively aggregate features in X by summing up their children's values.

    The method traverses the given Directed Acyclic Graph (DAG) hierarchy
    starting from the specified node, and recursively aggregates the values
    from its children nodes up to the specified root node. To caculate all
    values start form "ROOT".

    Parameters
    ----------
    X : {array-like, sparse matrix}
        The input array with the original data.
    hierarchy : networkx.DiGraph
            The Directed Acyclic Graph (DAG) representing the hierarchical
            structure.
    columns : list
            The mapping from the hierarchy graph's nodes to the columns in X.
            A list of ints. If this parameter is None the columns in X and
            the corresponding nodes in the hierarchy are expected to be in the
            same order.
    node : {int, str}
            The starting node for aggregation. Default is "ROOT".

    Returns
    ----------
    X : numpy.ndarray
        The input array `X` with the aggregated values based on the provided
        hierarchy.
    """
    if hierarchy.out_degree(node) == 0:
        return X
    children = hierarchy.successors(node)
    aggregated = np.zeros((X.shape[0]))
    for child in list(children):
        X = compute_aggregated_values(X, hierarchy, columns, node=child)
        aggregated = np.add(aggregated, X[:, columns.index(child)])

    if node != "ROOT":
        aggregated = np.add(aggregated, X[:, columns.index(node)])
        column_index = columns.index(node)
        X[:, column_index] = aggregated
    return X
