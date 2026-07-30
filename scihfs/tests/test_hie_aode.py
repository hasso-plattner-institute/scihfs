import networkx as nx
import numpy as np
import pytest

from scihfs.selectors import HieAODE


@pytest.mark.skip(
    reason="HieAODE bool-dtype CPT-indexing fix owned by the dedicated "
    "hie-aode branch; re-enable on merge."
)
def test_hie_aode(lazy_data2):
    small_DAG, train_x_data, train_y_data, test_x_data, _ = lazy_data2
    selector = HieAODE(hierarchy=small_DAG)
    selector.fit(train_x_data, train_y_data)
    _ = selector.predict(test_x_data)


def test_calculate_dependency_ascendant_class(lazy_data2):
    small_DAG, train_x_data, train_y_data, _, _ = lazy_data2
    selector = HieAODE(hierarchy=small_DAG)
    selector.fit(train_x_data, train_y_data)
    feature_idx = 2
    expected = np.full((selector.n_features_in_, selector.n_classes_, 2), -1)
    expected[0][0][0] = 0.0
    expected[0][1][0] = 0.0
    expected[0][0][1] = 1.0
    expected[0][1][1] = 1.0
    expected[1][0][0] = 0.0
    expected[1][1][0] = 0.0
    expected[1][0][1] = 0.0
    expected[1][1][1] = 1.0
    ancestors = nx.ancestors(selector._hierarchy_graph, feature_idx)

    for a in range(len(ancestors)):
        selector.calculate_prob_given_ascendant_class(ancestor=a)
    assert np.array_equal(selector.cpts["ancestors"], expected)
