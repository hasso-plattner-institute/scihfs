import numpy as np
import pytest
from scipy import sparse

from scihfs.selectors.hill_climbing import BottomUpSelector, TopDownSelector

# The `dataset_type="numerical"` path is disabled (so is its initialization),
# the two corresponding tests below are commented out.


@pytest.mark.parametrize(
    "data",
    ["data1", "data1_2"],
)
def test_top_down_selection(data, result_hill_selection_td, request):
    data = request.getfixturevalue(data)
    X, y, hierarchy, columns = data
    expected, support = result_hill_selection_td
    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


def test_bottom_up_selection(data1, result_hill_selection_bu):
    X, y, hierarchy, columns = data1
    expected, support, k = result_hill_selection_bu
    selector = BottomUpSelector(hierarchy, k=k)
    selector.fit(X, y, columns)
    X = selector.transform(X)
    assert np.array_equal(X, expected)

    support_mask = selector.get_support()
    assert np.array_equal(support_mask, support)


@pytest.mark.parametrize(
    "sparse_type", [sparse.csr_array, sparse.csr_matrix], ids=["csr_array", "csr_matrix"]
)
@pytest.mark.parametrize(
    "Selector, kwargs", [(TopDownSelector, {}), (BottomUpSelector, {"k": 3})]
)
def test_hill_climbing_selectors_accept_sparse_like_dense(
    data1, Selector, kwargs, sparse_type
):
    # compute_aggregated_values previously densified X unconditionally; this
    # pins that a sparse fit reproduces the dense fit's selection exactly.
    X, y, hierarchy, columns = data1

    dense = Selector(hierarchy, **kwargs).fit(X, y, columns)
    sparse_fit = Selector(hierarchy, **kwargs).fit(sparse_type(X), y, columns)

    assert np.array_equal(sparse_fit.get_support(), dense.get_support())

    sparse_transformed = sparse_fit.transform(sparse_type(X))
    if sparse.issparse(sparse_transformed):
        sparse_transformed = sparse_transformed.toarray()
    assert np.array_equal(sparse_transformed, dense.transform(X))


# Numerical input is currently not supported. Corresponding code is retained (but inactive) for future reintroduction.
# def test_bottom_up_selection_numerical(data1, result_hill_selection_bu):
#     X, y, hierarchy, columns = data1
#     expected, support, k = result_hill_selection_bu
#     selector = BottomUpSelector(hierarchy, k=k, dataset_type="numerical")
#     selector.fit(X, y, columns)
#     X = selector.transform(X)
#     assert np.array_equal(X, expected)
#
#     support_mask = selector.get_support()
#     assert np.array_equal(support_mask, support)


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

    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    score_matrix = selector._calculate_scores(X)

    assert np.array_equal(score_matrix, score_matrix_expected)


# Numerical input is currently not supported. Corresponding code is retained (but inactive) for future reintroduction.
# def test_calculate_scores_numerical(data_numerical, result_score_matrix_numerical):
#     X, y, hierarchy, columns = data_numerical
#     score_matrix_expected = result_score_matrix_numerical
#
#     selector = TopDownSelector(hierarchy, dataset_type="numerical")
#     selector.fit(X, y, columns)
#     score_matrix = selector._calculate_scores(X)
#
#     assert np.array_equal(score_matrix, score_matrix_expected)


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

    kwargs = {"k": 3} if Selector is BottomUpSelector else {}
    selector = Selector(hierarchy, **kwargs)
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


def test_calculate_similarity_rejects_root(data1):
    # feature_set reaching _calculate_similarity is contractually ROOT-free
    # (see select_and_return_features). If "ROOT" ever slipped in regardless,
    # fancy-indexing a numpy array with a mixed str/int list must fail loudly
    # rather than silently falling back to `arr[i, None]` semantics.
    X, y, hierarchy, columns = data1
    selector = BottomUpSelector(hierarchy, k=3)
    selector.fit(X, y, columns)

    with pytest.raises(IndexError):
        selector._calculate_similarity(0, 1, ["ROOT", columns[0]])


def test_calculate_fitness_function_td(
    data1, result_comparison_matrix_td1, result_fitness_funtion_td1
):
    X, y, hierarchy, columns = data1

    fitness_expected = result_fitness_funtion_td1

    selector = TopDownSelector(hierarchy)
    selector.fit(X, y, columns)
    fitness = selector._fitness_function(result_comparison_matrix_td1)

    assert np.array_equal(fitness, fitness_expected)


# --- BottomUpSelector: k must be a positive int -----------------------------


def test_k_rejects_zero(data1):
    X, y, hierarchy, columns = data1
    selector = BottomUpSelector(hierarchy, k=0)
    with pytest.raises(ValueError, match="k"):
        selector.fit(X, y, columns)


def test_k_rejects_non_integer(data1):
    X, y, hierarchy, columns = data1
    selector = BottomUpSelector(hierarchy, k=1.5)
    with pytest.raises(TypeError, match="k"):
        selector.fit(X, y, columns)
