"""Functions for importing and preprocessing data for experiments."""
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from kgextension.generator import direct_type_generator


def process_data(
    path: str = "hfs/data/sport_tweets_train.tsv", test_version: bool = False
):
    """Extends dbpedia data with types and creates the hierarchy.

    This functions is currently only meant for the sports tweets dataset.
    The data is saved in a .csv file and the hierarchy as an networkx.DiGraph
    in a .pickle file.

    Parameters
    ----------
    path : str
        The path to the original dataset.
    test_version: bool
        If True only a small subset of the data (10 samples) is processed.
        This is meant for testing purposes. Default is False.
    """
    if test_version:
        data = pd.read_csv(Path(f"{path.split('.')[0]}_testing.tsv"), sep="\t")
    else:
        data = pd.read_csv(Path(path), sep="\t")
        data = data[:600]

    extended_data = direct_type_generator(
        data,
        ["Dbpedia_URI_1", "Dbpedia_URI_2", "Dbpedia_URI_3"],
        hierarchy=True,
        caching=False,
    )
    graph = extended_data.attrs["hierarchy"]
    cleaned_data = extended_data[
        [column for column in extended_data.columns if "type" in column]
    ].copy()
    column_names_mapping = dict(
        zip(
            cleaned_data.columns,
            [column.split("_")[1] for column in cleaned_data.columns],
        )
    )
    cleaned_data.rename(column_names_mapping, inplace=True, axis=1)
    cleaned_data = cleaned_data.astype(float)
    cleaned_data["label"] = data["label"]

    version = ""
    if test_version:
        version = "_testing1"
    else:
        version = "_subset600"
    pickle.dump(
        graph, open(Path(f"{path.split('.')[0]}_hierarchy{version}.pickle"), "wb")
    )
    cleaned_data.to_csv(
        Path(f"{path.split('.')[0]}_with_hierarchy{version}.csv"), index=False
    )


def load_data(
    path: str = "hfs/data/sport_tweets_train.tsv", test_version: bool = False
) -> tuple[pd.DataFrame, nx.DiGraph, np.ndarray]:
    """Loads the preprocessed data and hierarchy.

    This functions is currently only meant for the sports tweets dataset.

    Parameters
    ----------
    path : str
        The path to the original dataset.
    test_version: bool
        If True only a small subset of the data (10 samples) is processed.
        This is meant for testing purposes. Default is False.

    Returns
    ----------
    (data, labels, hierachy) : (pd.DataFrame, np.ndarray, nx.DiGraph)
        The loaded data, the corresponding labels and the hierarchy graph.
    """
    version = ""
    if test_version:
        version = "_testing"
    else:
        version = "_subset300"
    hierarchy = pickle.load(
        open(Path(f"{path.split('.')[0]}_hierarchy{version}.pickle"), "rb")
    )
    data = pd.read_csv(Path(f"{path.split('.')[0]}_with_hierarchy{version}.csv"))
    labels = np.array(data["label"])
    data.drop("label", axis=1, inplace=True)
    return data, labels, hierarchy


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
