"""
Collection of helper methods for the feature selection algorithms.
"""

# import math
import warnings
from fractions import Fraction

import networkx as nx
import numpy as np
import scipy.sparse as sp
from networkx.algorithms.simple_paths import all_simple_paths
from sklearn.utils.multiclass import type_of_target


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


def check_bool_dtype(X):
    """Raise ValueError if ``X`` is not bool-dtype.

    Most scihfs estimators (preprocessor and selectors) operate on binary
    features. Enforcing bool-dtype at the boundary lets us keep the
    vectorized propagation and downstream selector code simple, and it
    surfaces a clear error before silent miscomputation can happen on
    numeric inputs. A future propagation_mode parameter will reopen this
    contract for sum-propagation on numeric data.
    """
    if X.dtype != np.bool_:
        raise ValueError(
            f"scihfs estimators require bool-dtype input. "
            f"Got dtype={X.dtype}. "
            f"If your data is binary, convert with X.astype(bool). "
            f"Non-binary (numeric) inputs are not yet supported - "
            f"see the sum-propagation roadmap for HFE workflows."
        )


def check_binary_target(y):
    """Raise ValueError unless ``y`` has a binary target encoding drawing
    from ``{0, 1}` (can also be encoded as boolean ``False`` and ``True``).
    """
    y_type = type_of_target(y, input_name="y")
    if y_type != "binary":
        raise ValueError(
            f"Only binary classification is supported. scihfs estimators "
            f"require a binary target; got y_type={y_type!r}."
        )
    labels = set(np.unique(y).tolist())
    if not labels <= {0, 1}:
        raise ValueError(
            f"scihfs estimators require the binary target to be labelled 0 and "
            f"1 (boolean False and True, respectively); got"
            f"classes={sorted(labels)!r}. Relabel the encodings before fitting."
        )


def _check_unique_column_mappings(columns):
    """Raise ValueError if any non-(-1) value appears more than once in columns.

    The ``columns`` variable is a list of integers doing the column->node mapping and can be supplied
    directly by the user or be auto-derived from the DataFrame feature names.
    Two equal non-(-1) entries mean two data columns map to the same hierarchy node, which is ill-defined (the orphan column marker -1 is exempt and may repeat).
    The values reported on failure are node positions (not DataFrame column names).
    """
    seen, duplicates = set(), set()
    for c in columns:
        if c == -1:
            continue
        (duplicates if c in seen else seen).add(c)
    if duplicates:
        raise ValueError(
            f"Duplicate column->node mappings detected: {sorted(duplicates)}. "
            f"Each entry in `columns` (except for the orphan column marker -1) "
            f"must map a data column to a unique hierarchy node."
            f"Suggested solution: If X was a DataFrame, check it for duplicate column names with "
            f"df.columns[df.columns.duplicated()]. "
            f"prior to feeding the dataset to the Estimator."
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


def shrink_dag(relevant_nodes: list, digraph: nx.DiGraph):
    """Remove nodes that cannot reach any relevant node identifier.

    A node is kept iff it is itself a node from the ``relevant_nodes`` list, an
    ancestor of one, or the virtual ``"ROOT"``. Every other node is a dead
    branch of the ontology (no corresponding data column and no descendant
    that has one) and is removed.

    Parameters
    ----------
    relevant_nodes : list
            A list containing node identifiers that are considered relevant
    digraph : networkx.DiGraph
            The Directed Acyclic Graph (DAG) from which dead-branch nodes
            will be removed.

    Returns
    ----------
    digraph : networkx.DiGraph
            The resulting DAG after removing all dead-branch nodes.

    Notes
    -----
    Currently still mutating ``digraph`` in place AND returning it. Temporarily kept here for backward compatibility; will be addressed in the near future.
    """
    useful = set(relevant_nodes) | {"ROOT"}
    for node in relevant_nodes:
        useful |= nx.ancestors(digraph, node)
    digraph.remove_nodes_from(set(digraph.nodes()) - useful)
    return digraph


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


# Numerical input is currently not supported and previous related code has been commented out throughout this repository.
# def normalize_score(score, max_value):
#     """Normalize the given score using logarithmic scaling and a maximum value.

#     Parameters
#     ----------
#     score : float or int
#             The score to be normalized.
#     max_value : float or int
#             The maximum of the scores in the corresponding row.

#     Returns
#     ----------
#     float or int : The normalized score after applying logarithmic scaling.
#     """
#     if score != 0:
#         score = math.log(1 + (score / max_value)) + 1
#     return score


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
    # Sparse input is accepted throughout the library, but
    # currently 'densified' here. To avoid excessive memory
    # allocation, add a dedicated sparse implementation here in
    # the future.
    if sp.issparse(X):
        X = X.toarray()
    # Note that we deliberately use uint32 (unsigned, min 0, max 4294967295), to keep memory footprint low and because feature counts are always positive integers.
    if X.dtype == np.bool_:
        X = X.astype(np.uint32)
    if hierarchy.out_degree(node) == 0:
        return X
    children = hierarchy.successors(node)
    aggregated = np.zeros((X.shape[0]), dtype=np.uint32)
    for child in list(children):
        X = compute_aggregated_values(X, hierarchy, columns, node=child)
        aggregated = np.add(aggregated, X[:, columns.index(child)])

    if node != "ROOT":
        aggregated = np.add(aggregated, X[:, columns.index(node)])
        column_index = columns.index(node)
        X[:, column_index] = aggregated
    return X
