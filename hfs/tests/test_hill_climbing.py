import numpy as np
import pytest

from hfs.selectors.hill_climbing import BottomUpSelector, TopDownSelector

from .fixtures.fixtures import (
    data1,
    data1_2,
    data2,
    data3,
    data_numerical,
    result_comparison_matrix_bu1,
    result_comparison_matrix_bu2,
    result_comparison_matrix_bu3,
    result_comparison_matrix_td1,
    result_fitness_funtion_bu1,
    result_fitness_funtion_td1,
    result_hill_selection_bu,
    result_hill_selection_td,
    result_score_matrix1,
    result_score_matrix2,
    result_score_matrix3,
    result_score_matrix_numerical,
)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_hill_selection_td()),
        (data1_2(), result_hill_selection_td()),
    ],
)
def test_top_down_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support = result
    selector = TopDownSelector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_hill_selection_bu()),
    ],
)
def test_bottom_up_selection(data, result):
    X, y, hierarchy, columns = data
    expected, support, k = result
    selector = BottomUpSelector(hierarchy, k=k, dataset_type="binary")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_hill_selection_bu()),
    ],
)
def test_bottom_up_selection_numerical(data, result):
    X, y, hierarchy, columns = data
    expected, support, k = result
    selector = BottomUpSelector(hierarchy, k=k, dataset_type="numerical")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        (data1(), result_score_matrix1()),
        (data2(), result_score_matrix2()),
        (data3(), result_score_matrix3()),
    ],
)
def test_calculate_scores(data, result):
    X, y, hierarchy, columns = data
    score_matrix_expected = result

    selector = TopDownSelector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    score_matrix = selector._calculate_scores(X)

    assert np.array_equal(score_matrix, score_matrix_expected)


@pytest.mark.parametrize(
    "data, result",
    [
        (data_numerical(), result_score_matrix_numerical()),
    ],
)
def test_calculate_scores_numerical(data, result):
    X, y, hierarchy, columns = data
    score_matrix_expected = result

    selector = TopDownSelector(hierarchy, dataset_type="numerical")
    selector.fit(X, y, columns)
    score_matrix = selector._calculate_scores(X)

    assert np.array_equal(score_matrix, score_matrix_expected)


@pytest.mark.parametrize(
    "data, result, Selector",
    [
        (data1(), result_comparison_matrix_td1(), TopDownSelector),
        (data1(), result_comparison_matrix_bu1(), BottomUpSelector),
        (data2(), result_comparison_matrix_bu2(), BottomUpSelector),
        (data3(), result_comparison_matrix_bu3(), BottomUpSelector),
    ],
)
def test_comparison_matrix(data, result, Selector):
    X, y, hierarchy, columns = data
    comparison_matrix_expected = result

    selector = Selector(hierarchy)
    selector.fit(X, y, columns)
    comparison_matrix = selector._comparison_matrix(columns)

    assert np.array_equal(comparison_matrix, comparison_matrix_expected)


@pytest.mark.parametrize(
    "data, comparison_matrix, result",
    [
        (
            data1(),
            result_comparison_matrix_bu1(),
            result_fitness_funtion_bu1(),
        ),
    ],
)
def test_calculate_fitness_function_bu(data, comparison_matrix, result):
    X, y, hierarchy, columns = data

    fitness_expected, k = result

    selector = BottomUpSelector(hierarchy, k=k)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(comparison_matrix)

    assert np.array_equal(fitness, fitness_expected)


@pytest.mark.parametrize(
    "data, comparison_matrix, result",
    [
        (
            data1(),
            result_comparison_matrix_td1(),
            result_fitness_funtion_td1(),
        )
    ],
)
def test_calculate_fitness_function_td(data, comparison_matrix, result):
    X, y, hierarchy, columns = data

    fitness_expected = result

    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(comparison_matrix)

    assert np.array_equal(fitness, fitness_expected)
