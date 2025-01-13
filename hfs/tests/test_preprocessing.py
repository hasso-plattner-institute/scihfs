import networkx as nx
import numpy as np
import pytest

from hfs.data_utils import create_mapping_columns_to_nodes, load_data
from hfs.helpers import get_columns_for_numpy_hierarchy
from hfs.preprocessing import HierarchicalPreprocessor


@pytest.mark.parametrize(
    "data",
    ["data1_preprocessing", "data2_preprocessing"],
)
def test_hierarchical_preprocessor(data, request):
    data = request.getfixturevalue(data)
    X, X_transformed, hierarchy, columns, hierarchy_expected = data

    preprocessor = HierarchicalPreprocessor(hierarchy)

    preprocessor.fit(X, columns=columns)
    assert preprocessor.is_fitted_
    X = preprocessor.transform(X)
    assert np.array_equal(X, X_transformed)
    hierarchy_transformed = preprocessor.get_hierarchy()
    assert np.array_equal(hierarchy_transformed, hierarchy_expected)


# TODO rename to test_fit and update to check all submethods included in fit?
def test_fit(data3_preprocessing):
    X, hierarchy, hierarchy_transformed, X_identifiers = data3_preprocessing
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns=X_identifiers)
    assert preprocessor.is_fitted_
    hierarchy = preprocessor.get_hierarchy()
    assert np.equal(hierarchy.all(), hierarchy_transformed.all())


def test_preprocessor_real_data():
    X, _, hierarchy = load_data(test_version=True)
    columns = create_mapping_columns_to_nodes(X, hierarchy)
    X = X.to_numpy()
    hierarchy = nx.to_numpy_array(hierarchy)
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns)
    X_transformed = preprocessor.transform(X)
    hierarchy_updated = preprocessor.get_hierarchy()
    columns_updated = preprocessor.get_columns()
    assert X_transformed.shape[1] == len(columns_updated)
    assert hierarchy_updated.shape[1] == X_transformed.shape[1]
    assert [
        col for col in columns_updated if col not in range(hierarchy_updated.shape[1])
    ] == []


def test_adjust_node_names():
    # [4, 5, 0, 1, 3] # original node names
    # [0, 1, 2, 3, 4] # node names after transformation to numpy array
    # [2, 3, -1, 4] # mapping

    # [0, 1, 2, 3, 4, 5] # updated nodes
    # [2, 3, 5, 4] # updated mapping (without deleting or renaming)

    # [2, 3, 4, 5] # updated nodes (with deletion)
    # [2, 3, 5, 4] # updated mapping (without deleting or renaming)

    # [0, 1, 2, 3] # renamed nodes
    # [0, 1, 3, 2] # renamed nodes mapping

    X = np.zeros((4, 4))
    edges = [(4, 5), (0, 1), (0, 3), (0, 4)]
    hierarchy = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    preprocessor = HierarchicalPreprocessor(hierarchy)
    preprocessor.fit(X, columns=columns)
    preprocessor.transform(X)
    updated_columns = preprocessor.get_columns()
    assert updated_columns == [0, 1, 3, 2]


def test_columns_not_in_hierarchy_raises_warning():
    hierarchy_graph = nx.DiGraph([(0, 1), (2, 1)])
    hierarchy = nx.to_numpy_array(hierarchy_graph)
    estimator = HierarchicalPreprocessor(hierarchy)
    X = [[0.42, 4.2, 0.42], [4, 2, 0.42]]
    with pytest.warns(UserWarning):
        estimator.fit(X)
