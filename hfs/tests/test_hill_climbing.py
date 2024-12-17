import numpy as np
import pytest

from hfs.selectors.hill_climbing import BottomUpSelector, TopDownSelector


@pytest.mark.parametrize(
    "data",
    ["data1", "data1_2"],
)
def test_top_down_selection(data, result_hill_selection_td, request):
    data = request.getfixturevalue(data)
    X, y, hierarchy, columns = data
    expected, support = result_hill_selection_td
    selector = TopDownSelector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


def test_bottom_up_selection(data1, result_hill_selection_bu):
    X, y, hierarchy, columns = data1
    expected, support, k = result_hill_selection_bu
    selector = BottomUpSelector(hierarchy, k=k, dataset_type="binary")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


def test_bottom_up_selection_numerical(data1, result_hill_selection_bu):
    X, y, hierarchy, columns = data1
    expected, support, k = result_hill_selection_bu
    selector = BottomUpSelector(hierarchy, k=k, dataset_type="numerical")
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "data, result",
    [
        ("data1", "result_score_matrix1"),
        ("data2", "result_score_matrix2"),
        ("data3", "result_score_matrix3"),
    ],
)
def test_calculate_scores(data, result, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    score_matrix_expected = result

    selector = TopDownSelector(hierarchy, dataset_type="binary")
    selector.fit(X, y, columns)
    score_matrix = selector._calculate_scores(X)

    assert np.array_equal(score_matrix, score_matrix_expected)


def test_calculate_scores_numerical(data_numerical, result_score_matrix_numerical):
    X, y, hierarchy, columns = data_numerical
    score_matrix_expected = result_score_matrix_numerical

    selector = TopDownSelector(hierarchy, dataset_type="numerical")
    selector.fit(X, y, columns)
    score_matrix = selector._calculate_scores(X)

    assert np.array_equal(score_matrix, score_matrix_expected)


@pytest.mark.parametrize(
    "data, result, Selector",
    [
        ("data1", "result_comparison_matrix_td1", TopDownSelector),
        ("data1", "result_comparison_matrix_bu1", BottomUpSelector),
        ("data2", "result_comparison_matrix_bu2", BottomUpSelector),
        ("data3", "result_comparison_matrix_bu3", BottomUpSelector),
    ],
)
def test_comparison_matrix(data, result, Selector, request):
    data = request.getfixturevalue(data)
    result = request.getfixturevalue(result)
    X, y, hierarchy, columns = data
    comparison_matrix_expected = result

    selector = Selector(hierarchy)
    selector.fit(X, y, columns)
    comparison_matrix = selector._comparison_matrix(columns)

    assert np.array_equal(comparison_matrix, comparison_matrix_expected)


def test_calculate_fitness_function_bu(
    data1, result_comparison_matrix_bu1, result_fitness_funtion_bu1
):
    X, y, hierarchy, columns = data1

    fitness_expected, k = result_fitness_funtion_bu1

    selector = BottomUpSelector(hierarchy, k=k)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(result_comparison_matrix_bu1)

    assert np.array_equal(fitness, fitness_expected)


def test_calculate_fitness_function_td(
    data1, result_comparison_matrix_td1, result_fitness_funtion_td1
):
    X, y, hierarchy, columns = data1

    fitness_expected = result_fitness_funtion_td1

    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(result_comparison_matrix_td1)

    assert np.array_equal(fitness, fitness_expected)
