import numpy as np
import pytest

from hfs import GASel
from hfs.selectors.gasel import _crossover, geometric_mean_sensitivity_specificity

@pytest.fixture
def simple_pred():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    return y_true, y_pred

@pytest.fixture
def zero_sensitivity_case():
    y_true = np.array([1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0])
    return y_true, y_pred


@pytest.fixture
def zero_specificity_case():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 1])
    return y_true, y_pred


@pytest.mark.parametrize(
    "data", ["zero_sensitivity_case", "zero_specificity_case"]
)
def test_geometric_mean_sensitivity_specificity(data, request):
    y_true, y_pred = request.getfixturevalue(data)
    score = geometric_mean_sensitivity_specificity(y_true, y_pred)
    assert isinstance(score, float)

@pytest.mark.parametrize(
    "data", ["zero_sensitivity_case", "zero_specificity_case"]
)
def test_zero_gm(data, request):
    y_true, y_pred = request.getfixturevalue(data)
    score = geometric_mean_sensitivity_specificity(y_true, y_pred)

    assert score == 0

def test_crossover():
    parent1 = np.array([0, 1, 0, 1])
    parent2 = np.array([0, 1, 1, 0])

    child1, child2 = _crossover(parent1, parent2)

    assert isinstance(child1, np.ndarray)
    assert isinstance(child2, np.ndarray)

def test_hierarchy():
    pass

