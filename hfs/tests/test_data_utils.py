import os
from pathlib import Path

import pytest

from hfs.data_utils import create_mapping_columns_to_nodes, load_data, process_data

from .fixtures.fixtures import (
    dataframe,
    hierarchy1,
    hierarchy1_2,
    hierarchy2,
    hierarchy3,
)


def remove_created_files():
    dirname = os.path.dirname(__file__)
    data_file = os.path.join(
        dirname, Path("../data/sport_tweets_train_with_hierarchy_testing1.csv")
    )
    hierarchy_file = os.path.join(
        dirname, Path("../data/sport_tweets_train_hierarchy_testing1.pickle")
    )
    os.remove(data_file)
    os.remove(hierarchy_file)


def test_process_data_no_error():
    process_data(test_version=True)
    remove_created_files()


def test_load_data():
    X, labels, hierarchy = load_data(test_version=True)
    mapping = create_mapping_columns_to_nodes(X, hierarchy)
    assert len(mapping) == 269
    assert len(X) == labels.shape[0]


@pytest.mark.parametrize(
    "hierarchy, dataframe",
    [
        (hierarchy1(), dataframe()),
        (hierarchy1_2(), dataframe()),
        (hierarchy2(), dataframe()),
        (hierarchy3(), dataframe()),
    ],
)
def test_create_mapping_columns_to_nodes(hierarchy, dataframe):
    mapping = create_mapping_columns_to_nodes(dataframe, hierarchy)
    nodes = list(hierarchy.nodes)
    for index, node in enumerate(dataframe.columns):
        assert nodes[mapping[index]] == node
