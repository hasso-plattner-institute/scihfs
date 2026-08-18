"""Behaviour tests for the lazy hierarchical feature-selection classifiers.

The golden masks and predictions below were captured from the pre-reshape
implementation on the real fit + predict path (no post-fit monkeypatching), so
they pin the algorithms' behaviour across the lazy-classifier redesign. The
``lazy_data2`` fixture carries real (non-degenerate) training data, so every
algorithm's relevance / MST branches are actually exercised there.
"""

import copy

import networkx as nx
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import BernoulliNB

from scihfs.metrics import mean_selected_fraction, sensitivity_specificity_product
from scihfs.selectors import HIP, HNB, MR, RNB, TAN, HieAODE, HNBs

# ---------------------------------------------------------------------------
# Golden masks + predictions on lazy_data2 (real training data).
# ---------------------------------------------------------------------------

_LAZY_DATA2_CASES = [
    (lambda h: HIP(h), [[0, 1, 1, 1], [0, 0, 1, 1]], [0, 1]),
    (lambda h: HNB(hierarchy=h, k=2), [[0, 1, 1, 0], [0, 0, 1, 1]], [0, 1]),
    (lambda h: HNBs(hierarchy=h), [[0, 1, 1, 1], [0, 0, 1, 1]], [0, 1]),
    (lambda h: RNB(hierarchy=h, k=2), [[0, 1, 1, 0], [0, 1, 1, 0]], [0, 1]),
    (lambda h: MR(h), [[0, 1, 1, 1], [0, 0, 1, 1]], [0, 1]),
    (lambda h: TAN(h), [[1, 1, 1, 1], [1, 1, 0, 0]], [0, 1]),
]


@pytest.mark.parametrize(
    "factory, exp_masks, exp_pred",
    _LAZY_DATA2_CASES,
    ids=["HIP", "HNB", "HNBs", "RNB", "MR", "TAN"],
)
def test_lazy_selectors_data2(lazy_data2, factory, exp_masks, exp_pred):
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = factory(small_DAG)
    assert selector.fit(X_train, y_train) is selector

    masks = selector.select(X_test)
    assert masks.dtype == bool
    assert masks.shape == (X_test.shape[0], X_train.shape[1])
    assert np.array_equal(masks.astype(int), np.array(exp_masks))

    preds = selector.predict(X_test)
    assert np.array_equal(preds, np.array(exp_pred))


@pytest.mark.filterwarnings("ignore:Hierarchy consists of multiple")
@pytest.mark.parametrize(
    "factory", [lambda h: HIP(h), lambda h: MR(h)], ids=["HIP", "MR"]
)
def test_lazy_selectors_data1(lazy_data1, factory):
    hierarchy, X_train, y_train, X_test, _, _ = lazy_data1
    selector = factory(nx.to_numpy_array(hierarchy)).fit(X_train, y_train)

    exp_masks = [[0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]] * 2
    assert np.array_equal(selector.select(X_test).astype(int), np.array(exp_masks))
    assert np.array_equal(selector.predict(X_test), np.array([0, 0]))


@pytest.mark.filterwarnings("ignore:Hierarchy consists of multiple")
def test_TAN_data3(lazy_data3):
    hierarchy, X_train_ones, _, y_train, X_test, _, _ = lazy_data3
    selector = TAN(nx.to_numpy_array(hierarchy)).fit(X_train_ones, y_train)

    # X_train_ones is constant, so every pairwise CMI is exactly 0 and MST
    # edge order is decided entirely by tie-breaking. _build_mst() now sorts
    # with kind="stable", so ties resolve by flat (node1, node2) index on
    # every platform; these golden masks were captured under that rule.
    exp_masks = [[1, 1, 1, 1, 0, 1], [1, 1, 0, 1, 1, 0]]
    assert np.array_equal(selector.select(X_test).astype(int), np.array(exp_masks))
    assert np.array_equal(selector.predict(X_test), np.array([0, 0]))


# ---------------------------------------------------------------------------
# Regression: a non-identity columns= mapping must be honoured.
#
# fit() relabels the hierarchy nodes to their data-column indices, so the
# per-instance status dicts are already keyed by column index. select() and
# _predict_instance() must therefore use the node directly; mapping it a second
# time through _column_index() (the former bug) is a no-op only when columns is
# the identity, so it went unnoticed. Here the data columns are permuted and a
# matching columns= mapping is supplied, so old feature k stays tied to node k
# but sits in a different column. Correct output is then equivariant under that
# permutation; the double-map is not.
#
# TAN is deliberately excluded: its MST tie-breaking (np.argsort over the CMI
# matrix) is itself column-order dependent, so equal-CMI edges make its output
# non-equivariant regardless of this bug. It shares the same (fixed) select /
# _predict_instance as the others, so nothing about the fix goes uncovered.
# ---------------------------------------------------------------------------

_PERM_FACTORIES = [
    lambda h: HIP(h),
    lambda h: HNB(hierarchy=h, k=2),
    lambda h: HNBs(hierarchy=h),
    lambda h: RNB(hierarchy=h, k=2),
    lambda h: MR(h),
]


@pytest.mark.parametrize(
    "factory", _PERM_FACTORIES, ids=["HIP", "HNB", "HNBs", "RNB", "MR"]
)
def test_lazy_selectors_honour_non_identity_columns(lazy_data2, factory):
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    # new column j holds old column perm[j], remapped back onto node perm[j].
    perm = [2, 0, 3, 1]

    base = factory(small_DAG).fit(X_train, y_train)  # identity columns
    permuted = factory(small_DAG).fit(X_train[:, perm], y_train, columns=perm)

    # predictions are per-instance labels -> unchanged by column reordering.
    assert np.array_equal(permuted.predict(X_test[:, perm]), base.predict(X_test))
    # masks follow the same permutation of the feature axis.
    assert np.array_equal(permuted.select(X_test[:, perm]), base.select(X_test)[:, perm])


# ---------------------------------------------------------------------------
# Sparse (CSR-bool) input must match dense input exactly.
#
# The lazy classifiers keep scipy.sparse input sparse end-to-end (only the
# instance currently being classified is densified), so fitting and predicting
# on a CSR-bool copy of the data must reproduce the dense predict / select
# exactly (same fit, same per-instance selection, same scoring). Both CSR
# containers (array and matrix) are covered.
#
# Equivalence alone would still hold if the classifiers densified internally,
# so the tests below it pin the memory behaviour itself.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sparse_type", [sp.csr_array, sp.csr_matrix], ids=["csr_array", "csr_matrix"]
)
@pytest.mark.parametrize(
    "factory",
    [case[0] for case in _LAZY_DATA2_CASES],
    ids=["HIP", "HNB", "HNBs", "RNB", "MR", "TAN"],
)
def test_lazy_selectors_accept_sparse_like_dense(lazy_data2, factory, sparse_type):
    small_DAG, X_train, y_train, X_test, _ = lazy_data2

    dense = factory(small_DAG).fit(X_train, y_train)
    sparse_fit = factory(small_DAG).fit(sparse_type(X_train), y_train)

    assert np.array_equal(sparse_fit.predict(sparse_type(X_test)), dense.predict(X_test))
    # select mirrors the input format, so sparse input returns a sparse mask;
    # densify it here to compare the selections themselves.
    sparse_masks = sparse_fit.select(sparse_type(X_test))
    assert sp.issparse(sparse_masks)
    assert np.array_equal(sparse_masks.toarray(), dense.select(X_test))


# The selectors that work on sparse X end-to-end. TAN and HieAODE are excluded
# on purpose: they densify for themselves (see the test below).
_SPARSE_NATIVE_FACTORIES = [
    lambda h: HIP(h),
    lambda h: HNB(hierarchy=h, k=2),
    lambda h: HNBs(hierarchy=h),
    lambda h: RNB(hierarchy=h, k=2),
    lambda h: MR(h),
]
_SPARSE_NATIVE_IDS = ["HIP", "HNB", "HNBs", "RNB", "MR"]


@pytest.mark.parametrize("factory", _SPARSE_NATIVE_FACTORIES, ids=_SPARSE_NATIVE_IDS)
def test_lazy_selectors_store_training_data_sparse(lazy_data2, factory):
    # The training matrix is retained for the estimator's lifetime, so
    # densifying it at fit time would cost O(n_samples * n_features) for as
    # long as the estimator lives. It must stay in its sparse form.
    small_DAG, X_train, y_train, _, _ = lazy_data2
    selector = factory(small_DAG).fit(sp.csr_array(X_train), y_train)
    assert sp.issparse(selector._xtrain)


@pytest.mark.parametrize(
    "factory", [lambda h: TAN(h), lambda h: HieAODE(h)], ids=["TAN", "HieAODE"]
)
def test_dense_only_selectors_densify_training_data_themselves(lazy_data2, factory):
    # TAN and HieAODE index columns/rows densely and carry dense
    # O(n_features**2) structures, so each densifies the training matrix in its
    # own _fit rather than the base class doing it for everyone. Pinning that
    # keeps it deliberate rather than accidental: when either is reworked, this
    # test is the reminder to drop its local .toarray().
    small_DAG, X_train, y_train, _, _ = lazy_data2
    selector = factory(small_DAG).fit(sp.csr_array(X_train), y_train)
    assert isinstance(selector._xtrain, np.ndarray)


@pytest.mark.parametrize("factory", _SPARSE_NATIVE_FACTORIES, ids=_SPARSE_NATIVE_IDS)
def test_lazy_selectors_select_on_dense_rows_of_sparse_input(lazy_data2, factory):
    # The per-instance algorithms index single features (x_row[node]), which a
    # sparse row slice does not support. Each instance must therefore arrive as
    # a dense 1-D array -- one row at a time, never the whole test matrix.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = factory(small_DAG).fit(sp.csr_array(X_train), y_train)

    seen_rows = []
    select_per_instance = selector._select_features_per_instance

    def record(x_row):
        seen_rows.append(x_row)
        return select_per_instance(x_row)

    selector._select_features_per_instance = record
    selector.predict(sp.csr_array(X_test))

    assert len(seen_rows) == X_test.shape[0]
    assert all(isinstance(row, np.ndarray) for row in seen_rows)
    assert all(row.shape == (X_train.shape[1],) for row in seen_rows)


# ---------------------------------------------------------------------------
# Selection masks mirror the input format.
#
# A mask of selected features is inherently sparse -- so on a large sparse
# problem a dense mask would have equal memory consumption than the data.
# Sparse X therefore yields a sparse mask, while dense X keeps returning an
# ndarray so existing callers are unaffected.
# The format follows the X handed to select/predict (not the one used in fit).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("factory", _SPARSE_NATIVE_FACTORIES, ids=_SPARSE_NATIVE_IDS)
def test_select_masks_mirror_input_format(lazy_data2, factory):
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    # Deliberately fitted on dense data: the mask format follows the test input.
    selector = factory(small_DAG).fit(X_train, y_train)

    dense_masks = selector.select(X_test)
    sparse_masks = selector.select(sp.csr_array(X_test))

    assert isinstance(dense_masks, np.ndarray)
    assert dense_masks.dtype == bool
    assert sp.issparse(sparse_masks)
    assert sparse_masks.dtype == bool
    assert sparse_masks.shape == dense_masks.shape
    assert np.array_equal(sparse_masks.toarray(), dense_masks)


@pytest.mark.parametrize("factory", _SPARSE_NATIVE_FACTORIES, ids=_SPARSE_NATIVE_IDS)
def test_predict_return_masks_mirror_input_format(lazy_data2, factory):
    # predict(return_masks=True) builds the masks in the same sweep as the
    # predictions, so it must follow the same rule as select().
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = factory(small_DAG).fit(sp.csr_array(X_train), y_train)

    _, sparse_masks = selector.predict(sp.csr_array(X_test), return_masks=True)
    _, dense_masks = selector.predict(X_test, return_masks=True)

    assert sp.issparse(sparse_masks)
    assert isinstance(dense_masks, np.ndarray)
    assert np.array_equal(sparse_masks.toarray(), dense_masks)
    # ... and both still agree with the standalone select().
    assert np.array_equal(dense_masks, selector.select(X_test))


def test_build_masks_handles_empty_selection(lazy_data2):
    # Nothing selected for any instance: the sparse branch then builds from
    # empty coordinate lists, which must still give a correctly shaped
    # all-False mask rather than a degenerate or 0x0 one.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HNB(hierarchy=small_DAG, k=2).fit(X_train, y_train)
    n_samples = X_test.shape[0]
    expected_shape = (n_samples, X_train.shape[1])

    dense_masks = selector._build_masks([], [], n_samples, as_sparse=False)
    sparse_masks = selector._build_masks([], [], n_samples, as_sparse=True)

    assert dense_masks.shape == expected_shape
    assert not dense_masks.any()
    assert sparse_masks.shape == expected_shape
    assert sparse_masks.count_nonzero() == 0


def test_mean_selected_fraction_matches_across_mask_formats(lazy_data2):
    # The metric is the main consumer of the masks and previously raised
    # ValueError on sparse input (np.asarray on a sparse matrix).
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HNB(hierarchy=small_DAG, k=2).fit(X_train, y_train)

    dense_fraction = mean_selected_fraction(selector.select(X_test))
    sparse_fraction = mean_selected_fraction(selector.select(sp.csr_array(X_test)))

    assert dense_fraction == sparse_fraction == 0.5


# ---------------------------------------------------------------------------
# DataFrame input: the column->node mapping is auto-derived from feature names.
#
# The lazy classifiers inherit HierarchyMixin's auto-derive plumbing: a
# DataFrame X + a named DiGraph + columns=None maps each feature name to its
# node (see _auto_derive_columns), so passing a DataFrame is exactly equivalent
# to passing the same array with the derived columns= mapping. The frame below
# lists its columns in shuffled order so a positional mapping would be wrong.
# ---------------------------------------------------------------------------


def _named_graph():
    return nx.DiGraph([("A", "B"), ("B", "C"), ("A", "D")])


def _named_dataframe():
    return pd.DataFrame(
        {
            "C": [1, 0, 0, 1],
            "A": [1, 1, 0, 1],
            "D": [0, 1, 0, 0],
            "B": [1, 0, 0, 1],
        }
    ).astype(bool)


_DF_Y = np.array([1, 0, 0, 1])


def test_lazy_autoderives_columns_from_dataframe():
    # Nodes A=0, B=1, C=2, D=3; frame columns C, A, D, B -> mapping [2, 0, 3, 1].
    selector = HIP(_named_graph()).fit(_named_dataframe(), _DF_Y)
    assert selector.get_columns() == [2, 0, 3, 1]


def test_lazy_ndarray_keeps_positional_mapping():
    # A plain ndarray has no feature names, so the mapping stays positional --
    # which is worth a warning here, because the node names say otherwise.
    X = _named_dataframe().to_numpy()
    with pytest.warns(UserWarning, match="by position"):
        selector = HIP(_named_graph()).fit(X, _DF_Y)
    assert not hasattr(selector, "feature_names_in_")
    assert selector.get_columns() == [0, 1, 2, 3]


@pytest.mark.parametrize(
    "factory",
    [case[0] for case in _LAZY_DATA2_CASES],
    ids=["HIP", "HNB", "HNBs", "RNB", "MR", "TAN"],
)
def test_lazy_dataframe_matches_explicit_columns(factory):
    # DataFrame input must be identical to the same array fed with the
    # auto-derived columns= mapping -- predict, select and the mapping itself.
    df = _named_dataframe()
    derived = [2, 0, 3, 1]

    df_fit = factory(_named_graph()).fit(df, _DF_Y)
    col_fit = factory(_named_graph()).fit(df.to_numpy(), _DF_Y, columns=derived)

    assert df_fit.get_columns() == derived
    assert np.array_equal(df_fit.predict(df), col_fit.predict(df.to_numpy()))
    assert np.array_equal(df_fit.select(df), col_fit.select(df.to_numpy()))


def test_lazy_orphan_dataframe_column_raises():
    # A DataFrame column with no matching node is rejected (selectors cannot
    # extend the hierarchy, unlike the HierarchicalPreprocessor).
    df = _named_dataframe()
    df["unicorn"] = [False, True, False, False]
    with pytest.raises(ValueError, match="no matching node"):
        HIP(_named_graph()).fit(df, _DF_Y)


def test_lazy_dataframe_with_adjacency_hierarchy_raises():
    # An adjacency-matrix hierarchy has no node names to match feature names.
    hierarchy = nx.to_numpy_array(_named_graph())
    with pytest.raises(ValueError, match="Cannot auto-derive columns"):
        HIP(hierarchy).fit(_named_dataframe(), _DF_Y)


# ---------------------------------------------------------------------------
# The hierarchy nodes and the data columns must be in bijection.
#
# The lazy classifiers work on a column-keyed graph (each node is a data
# column), so a hierarchy node with no column -- or a column with no node --
# is rejected up front rather than silently mis-indexed. (The preprocessor is
# the one estimator that tolerates and fixes such a mismatch instead.)
# ---------------------------------------------------------------------------


def test_lazy_selector_rejects_hierarchy_column_mismatch():
    X = np.array([[1, 1, 0, 1], [1, 0, 0, 0]], dtype=bool)  # 4 columns
    y = np.array([0, 1])

    # More nodes than columns: node 4 has no data column.
    too_many_nodes = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (2, 3), (3, 4)]))
    with pytest.raises(ValueError, match="not aligned"):
        HIP(too_many_nodes).fit(X, y)

    # More columns than nodes: column 3 has no hierarchy node.
    too_few_nodes = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2)]))
    with pytest.raises(ValueError, match="not aligned"):
        HIP(too_few_nodes).fit(X, y)


# ---------------------------------------------------------------------------
# Estimator contract: fitted-gating and prediction purity.
# ---------------------------------------------------------------------------


def test_predict_and_select_require_fit(lazy_data2):
    small_DAG, _, _, X_test, _ = lazy_data2
    selector = HIP(small_DAG)
    with pytest.raises(NotFittedError):
        selector.predict(X_test)
    with pytest.raises(NotFittedError):
        selector.select(X_test)


def test_predict_and_select_are_pure(lazy_data2):
    # predict/select must not mutate the fitted estimator's __dict__.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HNB(hierarchy=small_DAG, k=2).fit(X_train, y_train)
    before = copy.copy(selector.__dict__)

    selector.predict(X_test)
    selector.predict(X_test, return_masks=True)
    selector.select(X_test)

    assert selector.__dict__.keys() == before.keys()
    for key in before:
        assert selector.__dict__[key] is before[key]


def test_masked_nb_matches_bernoullinb_and_empty_is_majority(lazy_data2):
    # The one-shot masked NB must (a) reproduce a stock BernoulliNB's class
    # probabilities when every column is selected -- the equivalence that keeps
    # the golden predictions unchanged after dropping the fit-a-clone-per-instance
    # loop -- and (b) fall back to the training majority class on an empty
    # selection, since with no evidence the joint log-likelihood reduces to the
    # class log-prior.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HIP(small_DAG).fit(X_train, y_train)
    reference = BernoulliNB().fit(X_train, y_train)
    all_columns = list(range(X_train.shape[1]))
    majority = np.bincount(y_train).argmax()

    for row in X_test:
        masked = selector._nb.predict_proba_masked(row, all_columns)
        assert np.allclose(masked, reference.predict_proba(row.reshape(1, -1))[0])
        empty = selector._nb.predict_proba_masked(row, [])
        assert selector._nb.classes_[np.argmax(empty)] == majority


def test_predict_proba_normalized_and_consistent_with_predict(lazy_data2):
    # predict_proba is the primitive: each row is a proper distribution over
    # classes_, and predict is exactly its per-instance argmax.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HNB(hierarchy=small_DAG, k=2).fit(X_train, y_train)
    proba = selector.predict_proba(X_test)

    assert proba.shape == (X_test.shape[0], selector.classes_.shape[0])
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.array_equal(
        selector.classes_[np.argmax(proba, axis=1)], selector.predict(X_test)
    )


def test_predict_return_masks_matches_predict_and_select(lazy_data2):
    # predict(return_masks=True) yields both outputs from one sweep: the
    # predictions match plain predict and the masks match select exactly.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HNB(hierarchy=small_DAG, k=2).fit(X_train, y_train)

    preds, masks = selector.predict(X_test, return_masks=True)
    assert np.array_equal(preds, selector.predict(X_test))
    assert masks.dtype == bool
    assert np.array_equal(masks, selector.select(X_test))

    # The default stays single-output (backward compatible).
    plain = selector.predict(X_test)
    assert isinstance(plain, np.ndarray)


def test_hie_aode_disables_predict_proba(lazy_data2):
    # HieAODE overrides predict with AODE-style aggregation, so the inherited
    # naive-Bayes predict_proba would be silently inconsistent -- it is disabled.
    small_DAG, X_train, y_train, X_test, _ = lazy_data2
    selector = HieAODE(small_DAG).fit(X_train, y_train)
    with pytest.raises(AttributeError):
        selector.predict_proba(X_test)


# ---------------------------------------------------------------------------
# _validate_hyperparameters: no-op hook, called first thing in fit.
#
# Mirrors the equivalent eager-path tests in test_eager_base.py; the lazy
# fit() re-implements the template (it is not TransformerMixin-based), so the
# wiring is verified separately here.
# ---------------------------------------------------------------------------


class _RaisingHyperparameterHIP(HIP):
    """A stub whose hyperparameter validation always fails.

    Stands in for a future subclass that overrides the hook to validate its
    own hyperparameters; this repo has none yet, so the override is defined
    locally for the test.
    """

    def _validate_hyperparameters(self):
        raise ValueError("bad hyperparameter")


def test_lazy_validate_hyperparameters_default_is_a_noop():
    """The base HierarchyMixin implementation rejects nothing."""
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))
    X = np.zeros((2, 2), dtype=bool)
    selector = HIP(hierarchy).fit(X, np.array([0, 1]))
    assert selector.is_fitted_


def test_lazy_validate_hyperparameters_runs_before_hierarchy_none_check():
    """An invalid hyperparameter is reported even when hierarchy is also None.

    Hyperparameters are known from __init__ alone, so validating them first
    means this failure is reported instead of the (also true, but less
    specific, and only surfaced deep inside _fit_hierarchy on this path)
    "hierarchy is required" complaint.
    """
    X = np.zeros((2, 2), dtype=bool)
    selector = _RaisingHyperparameterHIP(None)
    with pytest.raises(ValueError, match="bad hyperparameter"):
        selector.fit(X, np.array([0, 1]))


def test_lazy_validate_hyperparameters_runs_before_data_validation():
    """An invalid hyperparameter is reported even when X is nonsense.

    X here is 1-D, which validate_data would reject with an unrelated
    "Expected 2D array" error -- proving the hyperparameter check runs first
    and validate_data is never reached.
    """
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))
    selector = _RaisingHyperparameterHIP(hierarchy)
    with pytest.raises(ValueError, match="bad hyperparameter"):
        selector.fit(np.zeros(4), np.array([0, 1]))


# ---------------------------------------------------------------------------
# Metrics helpers (former get_score internals, now in scihfs.metrics).
# ---------------------------------------------------------------------------


def test_lazy_metrics(lazy_data2):
    small_DAG, X_train, y_train, X_test, y_test = lazy_data2
    selector = HNB(hierarchy=small_DAG, k=2).fit(X_train, y_train)
    preds = selector.predict(X_test)
    masks = selector.select(X_test)

    # y_test = [1, 0], preds = [0, 1] -> both recalls 0 -> product 0.
    assert sensitivity_specificity_product(y_test, preds) == 0.0
    # masks [[0,1,1,0],[0,0,1,1]] -> 4 selected of 8 cells.
    assert mean_selected_fraction(masks) == 0.5
    # ClassifierMixin.score gives accuracy for free.
    assert selector.score(X_test, y_test) == 0.0
