import numpy as np
import pytest

from hfs.selectors import HieAODE

from .fixtures.fixtures import *


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_hie_aode(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    selector = HieAODE(hierarchy=small_DAG)
    selector.fit_selector(
        X_train=train_x_data, y_train=train_y_data, X_test=test_x_data
    )
    pred = selector.select_and_predict(predict=True, saveFeatures=True)


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_calculate_dependency_ascendant_class(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    selector = HieAODE(hierarchy=small_DAG)
    selector.fit_selector(
        X_train=train_x_data, y_train=train_y_data, X_test=test_x_data
    )
    sample_idx = 1
    sample = test_x_data[1]
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
    ancestors = nx.ancestors(selector._hierarchy, feature_idx)

    for a in range(len(ancestors)):
        selector.calculate_prob_given_ascendant_class(ancestor=a)
    assert np.array_equal(selector.cpts["ancestors"], expected)
