import numpy as np
import pytest

from hfs.selectors import HierTan

from .fixtures.fixtures import *


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_hiertan(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    selector = HierTan(hierarchy=small_DAG)
    selector.fit_selector(
        X_train=train_x_data, y_train=train_y_data, X_test=test_x_data
    )
    pred = selector.select_and_predict(predict=True, saveFeatures=True)
    assert(False)

@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_classification(data):
    assert(False)

@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_spanning_tree_training(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    selector = HierTan(hierarchy=small_DAG)
    selector.fit_selector(
        X_train=train_x_data, y_train=train_y_data, X_test=test_x_data
    )
    assert(False)

@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_probability_calculation_training(data):
    assert(False)
