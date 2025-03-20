"""Functions for importing and preprocessing data."""

import networkx as nx
import pandas as pd


def create_mapping_columns_to_nodes(data: pd.DataFrame, hierarchy: nx.DiGraph):
    """Creates a mapping from dataset columns to nodes in the hierarchy graph.

    For the estimators the hierarchy and the dataset will both be converted to
    numpy arrays and the column and node names will be lost. Therefore, a mapping
    to the corresponding indices is created so that after the transformation
    the correct nodes in the hierarchy can still be found for each column.

    Parameters
    ----------
    data : pd.Dataframe
        The dataset.
    hierarchy : nx.DiGraph
        The corresponding hierarchy.

    Returns
    ----------
    mapping : list
        A list of ints. The value at index i corresponds to the i'th column
        of the dataset. The value is the index of the corresponding node in
        the hierarchy.
    """
    columns = list(data.columns)
    nodes = list(hierarchy.nodes)
    mapping = [nodes.index(node) if node in nodes else -1 for node in columns]
    return mapping
