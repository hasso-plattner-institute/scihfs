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
    Calculate the relevance for a single node.

    Based on the column counts for feature present/absent,
    and how often presence/absence correlates with positive/negative
    target labels. Sparse is densified only per single column.

    Parameters
    ----------
    xdata : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
    ydata : array-like, shape (n_samples,)
            The target values. An array of int.
    node : int
        Node for which the relevance should be obtained.

    Returns
    ----------
    relevance : Fraction or int
        The relevance score of the node.
    """
    # List index slice covers ndarray, sparse matrices and sparse arrays
    # with a single function.
    column = xdata[:, [node]].toarray().ravel() if sp.issparse(xdata) else xdata[:, node]
    present = column == 1
    positive_target = np.asarray(ydata) == 1

    n_present = int(np.count_nonzero(present))
    n_absent = column.shape[0] - n_present
    n_present_positive = int(np.count_nonzero(present & positive_target))
    n_absent_positive = int(np.count_nonzero(positive_target)) - n_present_positive

    # Renamed the variables to match contingency table notation.
    # Cf. pX notation from the paper in the inline comments.
    ppv = Fraction(n_present_positive, n_present) if n_present else 0  # p1 = 1 - p3
    fomr = Fraction(n_absent_positive, n_absent) if n_absent else 0  # p2 = 1 - p4

    relevance = 2 * (ppv - fomr) ** 2
    return relevance


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


def check_square_adjacency_matrix(matrix):
    """Raise ValueError unless ``matrix`` is a 2-D square adjacency matrix.

    Checks conformance of the user's ndarray/scipy.sparse matrix
    input with adjacency matrix properties.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"The hierarchy adjacency matrix must be 2-dimensional, got "
            f"{matrix.ndim} dimension(s) with shape {matrix.shape}. Pass the "
            f"hierarchy as a square adjacency matrix (np.ndarray or "
            f"scipy.sparse) -- or directly as an nx.DiGraph."
        )
    n_rows, n_columns = matrix.shape
    if n_rows != n_columns:
        raise ValueError(
            f"The hierarchy adjacency matrix must be square, got shape "
            f"{matrix.shape}. Consider directly passing the hierarchy "
            f"as an nx.DiGraph instead of a matrix."
        )


def check_adjacency_matrix_values(matrix):
    """Raise ValueError if the adjacency matrix stores any edge
    information beyond mere presence.

    For scipy.sparse the explicitly stored values are checked.
    Converting back to a graph would create edges from all those
    explicitly stored values, even if they are zero.
    """
    if sp.issparse(matrix):
        stored = matrix.tocoo().data
        invalid = np.unique(stored[stored != 1])
        requirement = "every stored value has to be 1"
        hint = (
            " Any explicitly stored zero will be converted to an"
            "edge; drop these with matrix.eliminate_zeros()."
            if invalid.size and (invalid == 0).any()
            else ""
        )
    else:
        invalid = np.unique(matrix[(matrix != 0) & (matrix != 1)])
        requirement = "every entry has to be 0 (no edge) or 1 (edge)"
        hint = ""
    if invalid.size:
        raise ValueError(
            f"The hierarchy adjacency matrix must encode edge presence only, so "
            f"{requirement}; got {invalid[:5].tolist()}.{hint} Edge weights are "
            f"not supported."
        )


def check_digraph_edge_weights(digraph: nx.DiGraph):
    """Raise ValueError if a DiGraph edge carries a weight other than 1.

    Weightless edges are the encouraged format, ``weight=1`` is
    accepted as synonym. Any other weight information is ambiguous.
    """
    weighted = [
        (source, target, weight)
        for source, target, weight in digraph.edges(data="weight")
        if weight is not None and weight != 1
    ]
    if weighted:
        raise ValueError(
            f"Hierarchy edges must not carry a weight other than 1, but "
            f"{len(weighted)} edge(s) do: {weighted[:5]}. Edge weights are not "
            f"supported: an edge either exists or it does not. Drop the weight "
            f"attributes -- and for a weight of 0, drop the edge itself."
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

    Warns
    ----------
    UserWarning
                If the hierarchy falls apart into more than one connected
                component, i.e. it is a forest rather than a single hierarchy.
    """

    # Counted before "ROOT" is added and merges all components into a single one.
    n_components = nx.number_weakly_connected_components(hierarchy)
    # Roots do not equal multiple components, but still ALL require the adding of
    # another node.
    roots = [x for x in hierarchy.nodes() if hierarchy.in_degree(x) == 0]
    hierarchy.add_node("ROOT")
    if n_components > 1:
        warnings.warn(
            f"Hierarchy consists of multiple ({n_components}) disjoint hierarchies. "
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
    """Aggregate features in X by summing up their children's values/number
    of occurrences.

    The method traverses the given Directed Acyclic Graph (DAG) hierarchy
    starting from the specified node, and recursively aggregates the values
    from its children nodes up to the specified root node. To caculate all
    values start form "ROOT".

    Dispatcher-only: Sparse and dense input need different computation strategies.
    Those are implemented in the ``_compute_aggregated_values_{dense,sparse}`` functions.

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
    X : numpy.ndarray or scipy.sparse.csc_array
        The input array `X` with the aggregated values based on the provided
        hierarchy. Sparse input yields a sparse (csc_array) result; dense
        input yields a numpy.ndarray.
    """
    if sp.issparse(X):
        return _compute_aggregated_values_sparse(X, hierarchy, columns, node)
    return _compute_aggregated_values_dense(X, hierarchy, columns, node)


def _compute_aggregated_values_dense(X, hierarchy, columns, node="ROOT"):
    # Note that we deliberately use uint32 (unsigned, min 0, max 4294967295), to keep memory footprint low and because feature counts are always positive integers.
    if X.dtype == np.bool_:
        X = X.astype(np.uint32)
    # node -> column index, precomputed once so the recursion is O(1).
    column_of = {n: i for i, n in enumerate(columns)}
    return _aggregate_dense(X, hierarchy, column_of, node)


def _aggregate_dense(X, hierarchy, column_of, node):
    # Recursive aggregation of node values (occurrences) along the hierarchy,
    # dense format.
    if hierarchy.out_degree(node) == 0:
        return X
    aggregated = np.zeros((X.shape[0]), dtype=np.uint32)
    for child in list(hierarchy.successors(node)):
        X = _aggregate_dense(X, hierarchy, column_of, child)
        aggregated = np.add(aggregated, X[:, column_of[child]])

    if node != "ROOT":
        aggregated = np.add(aggregated, X[:, column_of[node]])
        X[:, column_of[node]] = aggregated
    return X


def _compute_aggregated_values_sparse(X, hierarchy, columns, node="ROOT"):
    X = X.tocsc()
    column_of = {n: i for i, n in enumerate(columns)}
    cache = {}
    _aggregate_sparse(X, hierarchy, column_of, node, cache)
    return _assemble_sparse(X, cache)


def _aggregate_sparse(X, hierarchy, column_of, node, cache):
    # Recursive aggregation of node values (occurrences) along the hierarchy,
    # sparse format.
    # Overwriting in-place is not cheap for CSC, so a transient dense vector
    # is written to 'cache' (internal non-leaf nodes only) before assembling
    # a sparse matrix once at the end.
    if hierarchy.out_degree(node) == 0:
        return
    aggregated = np.zeros(X.shape[0], dtype=np.uint32)
    for child in list(hierarchy.successors(node)):
        _aggregate_sparse(X, hierarchy, column_of, child, cache)
        aggregated = np.add(aggregated, _sparse_column_uint32(X, column_of, child, cache))

    if node != "ROOT":
        aggregated = np.add(aggregated, _sparse_column_uint32(X, column_of, node, cache))
        cache[column_of[node]] = aggregated


def _sparse_column_uint32(X, column_of, node, cache):
    # Returns a column with values converted to uint32 to allow summation
    # (and the pre-aggregated value if the column had already been visited
    # in a previous pass).
    index = column_of[node]
    if index in cache:
        return cache[index]
    column = X[:, index].toarray().ravel()
    return column.astype(np.uint32) if column.dtype == np.bool_ else column


def _assemble_sparse(X, cache):
    # Reassembles the sparse matrix (with uint32 values) from the cache
    # and the original (leaf) columns of X.
    # Added a cache.pop to reduce peak memory usage when appending to the
    # blocks variable.
    n_features = X.shape[1]
    blocks = []
    for index in range(n_features):
        if index in cache:  # already aggregated (non-leaf) nodes/features
            blocks.append(sp.csc_array(cache.pop(index).reshape(-1, 1)))
        else:  # original (leaf) nodes/features
            column = X[:, [index]]
            blocks.append(
                column.astype(np.uint32) if column.dtype == np.bool_ else column
            )
    return sp.hstack(blocks, format="csc")
