"""sklearn conformance tests for the scihfs estimators, semi-automatically
generated.

Two complementary test sections, both following the "check x estimator"
matrix pattern mirroring the shape of sklearn's parametrize_with_checks.

Section 1: Core sklearn conformance via check_estimator.

    ``test_all_estimators`` runs sklearn's ``check_estimator``. The suite
    partially feeds non-binary float/sparse data, which the current strict
    bool-dtype contract rejects. Those tests should not fail the entire suite,
    thus are listed under the ``_EXPECTED_FAILED_CHECKS`` and are exempted
    from the pass/fail logic.
    The remaining checks are dtype-independent and should pass as expected.

Section 2: Sklearn-adjacent conformance.

    Addresses those ``check_estimator`` checks that fail because of the
    current strict bool-dtype enforcement.

    Section 2A:
        Reimplements the corresponding sklearn checks with identical logic, but
        changes the input to random bool data (instead of random non-bool data).

    Section 2B:
        'Best effort' to reimplement the corresponding sklearn checks' logic, but
        required adaptation in the logic to fit the bool-dtype data format
        (or a bool-applicable subset of the original logic).

    Section 2C:
        Stubs ('pass') for those sklearn checks that fundamentally require
        different data and cannot be faithfully recreated.
"""

import pickle
import re
from copy import deepcopy
from functools import partial
from inspect import signature

import joblib
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sparse
from sklearn.base import clone, is_classifier
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.utils import _safe_indexing, get_tags
from sklearn.utils._testing import (
    assert_allclose,
    assert_allclose_dense_sparse,
    create_memmap_backed_data,
    set_random_state,
)
from sklearn.utils.estimator_checks import (
    _apply_on_subsets,
    _enforce_estimator_tags_X,
    _enforce_estimator_tags_y,
    _is_public_parameter,
    _NotAnArray,
    check_estimator,
)
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import check_is_fitted

from scihfs.tests._estimators import ALL_ESTIMATORS

# ---------------------------------------------------------------------------
# 1. sklearn check_estimator with the contract-incompatible checks declared.
# ---------------------------------------------------------------------------

# Shared reason for the expected-failed checks: every scihfs estimator now
# enforces bool-dtype input, but sklearn's check_estimator suite feeds
# numeric (float) / sparse arrays, so the dtype-dependent checks are expected to
# fail. Re-opens once numeric input will be supported.
_BOOL_DTYPE_XFAIL_REASON = (
    "enforces bool-dtype input; sklearn check_estimator feeds numeric arrays."
)

# The exact set of check_estimator checks that fail because scihfs enforces
# bool-dtype input. Declaring them here (instead of strict-xfailing the whole
# row) lets the dtype-independent conformance checks actually run and pass,
# while these are recorded as expected failures. With on_fail='raise' (the
# default), any NEW failure outside this set fails the test as a real regression.
#
# sklearn runs a role-specific subset of checks, so the expected failures are
# split accordingly: a shared core, plus the transformer-only checks (eager
# selectors + preprocessor) or the classifier-only checks (lazy selectors).
_COMMON_BOOL_XFAILS = (
    "check_dict_unchanged",
    "check_dont_overwrite_parameters",
    "check_dtype_object",
    "check_estimator_sparse_array",
    "check_estimator_sparse_matrix",
    "check_estimator_sparse_tag",
    "check_estimators_dtypes",
    "check_estimators_fit_returns_self",
    "check_estimators_nan_inf",
    "check_estimators_overwrite_params",
    "check_estimators_pickle",
    "check_f_contiguous_array_estimator",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_fit2d_predict1d",
    "check_fit_check_is_fitted",
    "check_fit_idempotent",
    "check_fit_score_takes_y",
    "check_methods_sample_order_invariance",
    "check_methods_subset_invariance",
    "check_n_features_in",
    "check_n_features_in_after_fitting",
    "check_pipeline_consistency",
    "check_positive_only_tag_during_fit",
    "check_readonly_memmap_input",
)


_TRANSFORMER_BOOL_XFAILS = (
    "check_transformer_data_not_an_array",
    "check_transformer_general",
    "check_transformer_preserve_dtypes",
)


_CLASSIFIER_BOOL_XFAILS = (
    "check_classifier_data_not_an_array",
    "check_classifiers_classes",
    "check_classifiers_one_label",
    "check_classifiers_train",
    "check_supervised_y_2d",
)


def _expected_failed_checks(estimator):
    role_xfails = (
        _CLASSIFIER_BOOL_XFAILS if is_classifier(estimator) else _TRANSFORMER_BOOL_XFAILS
    )
    return {name: _BOOL_DTYPE_XFAIL_REASON for name in _COMMON_BOOL_XFAILS + role_xfails}


@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_all_estimators(estimator):
    hierarchy_graph = nx.DiGraph()
    adj_matrix = nx.to_numpy_array(hierarchy_graph)
    est = estimator(adj_matrix)
    check_estimator(est, expected_failed_checks=_expected_failed_checks(est))


# ---------------------------------------------------------------------------
# 2. Positive conformance on contract-compliant (bool) data.
#
# These re-implement individual check_estimator checks (sklearn names + logic)
# but feed bool data so the behaviour runs and passes. Three groups:
#
#   Section 2A  -- dtype-independent invariants whose sklearn logic runs
#                 essentially verbatim once bool data is supplied; they fail
#                 under check_estimator only because the float data is rejected
#                 at the bool-dtype gate. Covers the generic invariants plus the
#                 shape edge cases, is-fitted, fit/score signature, sample-order
#                 / subset invariance, pipeline consistency, list input and
#                 dict-unchanged. The only deviations are the input data (bool)
#                 and its width (matched to the hierarchy -- a couple of checks
#                 that fix the feature count use a smaller hierarchy).
#   Section 2B -- checks that probe a specific input *form* or data property
#                 (sparse container, readonly memmap, F-contiguous layout,
#                 non-finite values, dtype preservation, transformer
#                 fit_transform consistency).
#   Section 2C -- checks whose intent fundamentally requires non-bool input
#                 (object dtype, multiple numeric dtypes, negative values), so
#                 they cannot run under the contract. Kept as name + ``pass``
#                 stubs.
# ---------------------------------------------------------------------------

_HIERARCHY = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (0, 3)]))
_N_FEATURES = _HIERARCHY.shape[0]


def _binary_Xy(n_samples, random_state=0):
    """Bool X (width-matched to the hierarchy) and a binary y.

    The bool/width-matched stand-in for the float data sklearn's checks
    generate: same role, but contract-compliant and shaped to the hierarchy.
    """
    rng = np.random.RandomState(random_state)
    X = rng.randint(0, 2, size=(n_samples, _N_FEATURES)).astype(bool)
    y = rng.randint(0, 2, size=n_samples)
    return X, y


def _fit_passes_or_matches(estimator, X, y, patterns):
    """Mirror sklearn's raises(ValueError, match=patterns, may_pass=True).

    fit either succeeds, or raises a ValueError whose message matches one of the
    patterns; any other outcome fails.
    """
    try:
        estimator.fit(X, y)
    except ValueError as e:
        assert any(
            re.search(p, str(e)) for p in patterns
        ), f"fit raised a ValueError not matching {patterns}: {e}"


# ---------------------------------------------------------------------------
# Section 2A:
# Reimplements the corresponding sklearn checks with identical logic, but
# changes the input to random bool data (instead of random non-bool data).
# ---------------------------------------------------------------------------


def check_estimators_fit_returns_self(name, estimator_orig):
    """Check if self is returned when calling fit."""
    X, y = _binary_Xy(n_samples=21)
    X = _enforce_estimator_tags_X(estimator_orig, X)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator


def check_n_features_in(name, estimator_orig):
    # Make sure that n_features_in_ doesn't exist until fit is called, and that
    # its value is correct.
    estimator = clone(estimator_orig)
    set_random_state(estimator)

    X, y = _binary_Xy(n_samples=100)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    assert not hasattr(estimator, "n_features_in_")
    estimator.fit(X, y)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == X.shape[1]


def check_n_features_in_after_fitting(name, estimator_orig):
    # Make sure n_features_in_ is set after fitting and that the prediction
    # methods reject inputs with the wrong number of features.
    tags = get_tags(estimator_orig)
    is_supported_X_types = tags.input_tags.two_d_array or tags.input_tags.categorical
    if not is_supported_X_types or tags.no_validation:
        return  # pragma: no cover

    estimator = clone(estimator_orig)
    set_random_state(estimator)

    X, y = _binary_Xy(n_samples=15)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    estimator.fit(X, y)
    assert hasattr(estimator, "n_features_in_")
    assert estimator.n_features_in_ == X.shape[1]

    check_methods = [
        "predict",
        "transform",
        "decision_function",
        "predict_proba",
        "score",
    ]
    X_bad = X[:, [1]]
    msg = f"X has 1 features, but \\w+ is expecting {X.shape[1]} features as input"
    for method in check_methods:
        if not hasattr(estimator, method):
            continue
        callable_method = getattr(estimator, method)
        if method == "score":
            callable_method = partial(callable_method, y=y)  # pragma: no cover
        with pytest.raises(ValueError, match=msg):
            callable_method(X_bad)


def check_estimators_pickle(name, estimator_orig):
    """Test that we can pickle all estimators."""
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    X, y = _binary_Xy(n_samples=30)
    X = _enforce_estimator_tags_X(estimator_orig, X)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)
    estimator.fit(X, y)

    unpickled_estimator = pickle.loads(pickle.dumps(estimator))

    result = {
        method: getattr(estimator, method)(X)
        for method in check_methods
        if hasattr(estimator, method)
    }
    for method in result:
        unpickled_result = getattr(unpickled_estimator, method)(X)
        assert_allclose_dense_sparse(result[method], unpickled_result)


def check_fit_idempotent(name, estimator_orig):
    # Check that est.fit(X).transform(X) is the same as est.fit(X).fit(X)
    # .transform(X), via the public methods.
    check_methods = ["predict", "transform", "decision_function", "predict_proba"]

    estimator = clone(estimator_orig)
    set_random_state(estimator)

    X, y = _binary_Xy(n_samples=100)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    rng = np.random.RandomState(0)
    train, test = next(ShuffleSplit(test_size=0.2, random_state=rng).split(X))
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, _ = _safe_split(estimator, X, y, test, train)

    # Fit for the first time
    estimator.fit(X_train, y_train)

    result = {
        method: getattr(estimator, method)(X_test)
        for method in check_methods
        if hasattr(estimator, method)
    }

    # Fit again
    set_random_state(estimator)
    estimator.fit(X_train, y_train)

    for method in check_methods:
        if hasattr(estimator, method):
            new_result = getattr(estimator, method)(X_test)
            if hasattr(new_result, "dtype") and np.issubdtype(
                new_result.dtype, np.floating
            ):
                tol = 2 * np.finfo(new_result.dtype).eps  # pragma: no cover
            else:
                tol = 2 * np.finfo(np.float64).eps
            assert_allclose_dense_sparse(
                result[method],
                new_result,
                atol=max(tol, 1e-9),
                rtol=max(tol, 1e-7),
                err_msg=f"Idempotency check failed for method {method}",
            )


def check_dont_overwrite_parameters(name, estimator_orig):
    # check that fit method only changes or sets private attributes
    estimator = clone(estimator_orig)
    X, y = _binary_Xy(n_samples=20)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator, 1)
    dict_before_fit = estimator.__dict__.copy()
    estimator.fit(X, y)

    dict_after_fit = estimator.__dict__

    public_keys_after_fit = [
        key for key in dict_after_fit.keys() if _is_public_parameter(key)
    ]

    attrs_added_by_fit = [
        key for key in public_keys_after_fit if key not in dict_before_fit.keys()
    ]
    assert not attrs_added_by_fit, (
        "Estimator adds public attribute(s) during the fit method. Estimators"
        " are only allowed to add private attributes either started with _ or"
        " ended with _ but %s added" % ", ".join(attrs_added_by_fit)
    )

    attrs_changed_by_fit = [
        key
        for key in public_keys_after_fit
        if (dict_before_fit[key] is not dict_after_fit[key])
    ]
    assert not attrs_changed_by_fit, (
        "Estimator changes public attribute(s) during the fit method. Estimators"
        " are only allowed to change attributes started or ended with _, but"
        " %s changed" % ", ".join(attrs_changed_by_fit)
    )


def check_estimators_overwrite_params(name, estimator_orig):
    X, y = _binary_Xy(n_samples=21)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    set_random_state(estimator)

    # Make a physical copy of the original estimator parameters before fitting.
    params = estimator.get_params()
    original_params = deepcopy(params)

    estimator.fit(X, y)

    # Compare the state of the model parameters with the original parameters
    new_params = estimator.get_params()
    for param_name, original_value in original_params.items():
        new_value = new_params[param_name]
        assert joblib.hash(new_value) == joblib.hash(original_value), (
            "Estimator %s should not change or mutate the parameter %s from %s"
            " to %s during fit." % (name, param_name, original_value, new_value)
        )


def check_fit2d_1feature(name, estimator_orig):
    # Fitting a 2d array with a single feature works or gives an informative
    # message. scihfs ties the feature count to the hierarchy, so the 1-feature
    # fit is run against a width-matched 1-node hierarchy.
    rnd = np.random.RandomState(0)
    X = rnd.randint(0, 2, size=(10, 1)).astype(bool)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    estimator = type(estimator_orig)(np.zeros((1, 1)))
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)

    msgs = [r"1 feature\(s\)", "n_features = 1", "n_features=1"]
    _fit_passes_or_matches(estimator, X, y, msgs)


def check_fit2d_1sample(name, estimator_orig):
    # Fitting a 2d array with a single sample works or gives an informative
    # message about the number of samples / classes.
    rnd = np.random.RandomState(0)
    X = rnd.randint(0, 2, size=(1, _N_FEATURES)).astype(bool)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    y = X[:, 0].astype(int)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)

    msgs = [
        "1 sample",
        "n_samples = 1",
        "n_samples=1",
        "one sample",
        "1 class",
        "one class",
    ]
    _fit_passes_or_matches(estimator, X, y, msgs)


def check_fit2d_predict1d(name, estimator_orig):
    # Fit a 2d array, then call the prediction methods with a 1d array.
    X, y = _binary_Xy(n_samples=20)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            with pytest.raises(ValueError, match="Reshape your data"):
                getattr(estimator, method)(X[0])


def check_fit_check_is_fitted(name, estimator_orig):
    # check_is_fitted must fail before fit and pass after.
    estimator = clone(estimator_orig)
    set_random_state(estimator)
    X, y = _binary_Xy(n_samples=100)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)

    if get_tags(estimator).requires_fit:
        with pytest.raises(NotFittedError):
            check_is_fitted(estimator)
    estimator.fit(X, y)
    check_is_fitted(estimator)  # must not raise


def check_fit_score_takes_y(name, estimator_orig):
    # All estimators must accept an optional y in fit/score so they compose in
    # pipelines.
    X, y = _binary_Xy(n_samples=30)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)

    funcs = ["fit", "score", "partial_fit", "fit_predict", "fit_transform"]
    for func_name in funcs:
        func = getattr(estimator, func_name, None)
        if func is not None:
            func(X, y)
            args = [p.name for p in signature(func).parameters.values()]
            if args[0] == "self":
                args = args[1:]  # pragma: no cover
            assert args[1] in [
                "y",
                "Y",
            ], "Expected y or Y as second argument for method %s of %s. Got %r." % (
                func_name,
                type(estimator).__name__,
                args,
            )


def check_methods_sample_order_invariance(name, estimator_orig):
    # Methods give invariant results under a permutation of the sample order.
    X, y = _binary_Xy(n_samples=20)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)
    estimator.fit(X, y)

    idx = np.random.permutation(X.shape[0])
    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            assert_allclose_dense_sparse(
                _safe_indexing(getattr(estimator, method)(X), idx),
                getattr(estimator, method)(_safe_indexing(X, idx)),
                atol=1e-9,
                err_msg=f"{method} of {name} is not sample-order invariant.",
            )


def check_methods_subset_invariance(name, estimator_orig):
    # Methods give invariant results on a subset vs the whole set.
    X, y = _binary_Xy(n_samples=20)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            result_full, result_by_batch = _apply_on_subsets(
                getattr(estimator, method), X
            )
            assert_allclose(
                result_full,
                result_by_batch,
                atol=1e-7,
                err_msg=f"{method} of {name} is not subset invariant.",
            )


def check_pipeline_consistency(name, estimator_orig):
    # make_pipeline(est) gives the same result as est on its own.
    if get_tags(estimator_orig).non_deterministic:
        pytest.skip(name + " is non deterministic")  # pragma: no cover

    X, y = _binary_Xy(n_samples=30)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)
    pipeline = make_pipeline(estimator)
    estimator.fit(X, y)
    pipeline.fit(X, y)

    for func_name in ["score", "fit_transform"]:
        func = getattr(estimator, func_name, None)
        if func is not None:
            result = func(X, y)
            result_pipe = getattr(pipeline, func_name)(X, y)
            assert_allclose_dense_sparse(result, result_pipe)


def check_transformer_data_not_an_array(name, transformer_orig):
    # The transformer behaves the same when X is not an ndarray (a _NotAnArray
    # wrapper, or a plain list). Classifier-only estimators have no transform.
    if not hasattr(transformer_orig, "transform"):
        return
    X, y = _binary_Xy(n_samples=30)
    X = _enforce_estimator_tags_X(transformer_orig, X)
    _check_transformer_binary(
        name, transformer_orig, _NotAnArray(X), _NotAnArray(np.asarray(y))
    )
    _check_transformer_binary(name, transformer_orig, X.tolist(), y.tolist())


def check_dict_unchanged(name, estimator_orig):
    # The prediction methods must not mutate the estimator's __dict__.
    X, y = _binary_Xy(n_samples=20)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator, 1)
    estimator.fit(X, y)

    for method in ["predict", "transform", "decision_function", "predict_proba"]:
        if hasattr(estimator, method):
            dict_before = estimator.__dict__.copy()
            getattr(estimator, method)(X)
            assert (
                estimator.__dict__ == dict_before
            ), f"Estimator changes __dict__ during {method}"


_SECTION_A_CHECKS = [
    check_estimators_fit_returns_self,
    check_n_features_in,
    check_n_features_in_after_fitting,
    check_estimators_pickle,
    check_fit_idempotent,
    check_dont_overwrite_parameters,
    check_estimators_overwrite_params,
    check_fit2d_1feature,
    check_fit2d_1sample,
    check_fit2d_predict1d,
    check_fit_check_is_fitted,
    check_fit_score_takes_y,
    check_methods_sample_order_invariance,
    check_methods_subset_invariance,
    check_pipeline_consistency,
    check_transformer_data_not_an_array,
    check_dict_unchanged,
]


# ---------------------------------------------------------------------------
# Section 2B:
# 'Best effort' to reimplement the corresponding sklearn checks' logic, but
# required adaptation in the logic to fit the bool-dtype data format (or a
# bool-applicable subset of the original logic).
# ---------------------------------------------------------------------------


def _check_estimator_sparse_container(name, estimator_orig, sparse_type):
    """Mirror sklearn's sparse-container check on bool data.

    Both the eager transformers and the lazy classifiers declare
    input_tags.sparse=True and validate with accept_sparse, so fitting sparse
    *bool* input must succeed -- the eager path keeps it sparse, the lazy path
    densifies internally (for now, might change in the future).
    (sklearn's own check feeds sparse *float*, which the bool-dtype contract
    rejects -- see check_estimator_sparse_array in the expected-failure sets.)
    The early return keeps the branch faithful to sklearn for any hypothetical
    sparse=False estimator, though no scihfs estimator declares that.
    """
    if not get_tags(estimator_orig).input_tags.sparse:
        return
    X, y = _binary_Xy(n_samples=40)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    X = sparse_type(X)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)
    set_random_state(estimator)

    estimator.fit(X, y)
    if hasattr(estimator, "predict"):
        assert estimator.predict(X).shape[0] == X.shape[0]  # pragma: no cover
    if hasattr(estimator, "transform"):
        assert estimator.transform(X).shape[0] == X.shape[0]


def check_estimator_sparse_array(name, estimator_orig):
    _check_estimator_sparse_container(name, estimator_orig, sparse.csr_array)


def check_estimator_sparse_matrix(name, estimator_orig):
    _check_estimator_sparse_container(name, estimator_orig, sparse.csr_matrix)


def check_estimator_sparse_tag(name, estimator_orig):
    """Check the input_tags.sparse tag is consistent with fit behaviour."""
    estimator = clone(estimator_orig)

    X, y = _binary_Xy(n_samples=40)
    X = _enforce_estimator_tags_X(estimator, X)
    y = _enforce_estimator_tags_y(estimator, y)
    X = sparse.csr_array(X)

    tags = get_tags(estimator)
    if tags.input_tags.sparse:
        try:
            estimator.fit(X, y)  # should pass
        except Exception as e:  # pragma: no cover
            raise AssertionError(
                f"Estimator {name} raised an exception. The tag "
                f"self.input_tags.sparse={tags.input_tags.sparse} might not be "
                "consistent with the estimator's ability to handle sparse data."
            ) from e
    else:  # pragma: no cover
        # No scihfs estimator declares sparse=False, but keep the branch
        # faithful to sklearn: it must then reject sparse with a clear message.
        try:
            estimator.fit(X, y)
        except (ValueError, TypeError) as e:
            if re.search("[Ss]parse", str(e)):
                return
            raise AssertionError(
                f"Estimator {name} failed on sparse data but the error did not "
                "state that sparse input is unsupported."
            ) from e
        raise AssertionError(
            f"Estimator {name} did not fail on sparse data despite "
            f"self.input_tags.sparse={tags.input_tags.sparse}."
        )


def check_readonly_memmap_input(name, estimator_orig):
    """Check that the estimator can handle readonly memmap backed data."""
    X, y = _binary_Xy(n_samples=21)
    X = _enforce_estimator_tags_X(estimator_orig, X)

    estimator = clone(estimator_orig)
    y = _enforce_estimator_tags_y(estimator, y)

    X, y = create_memmap_backed_data([X, y])

    set_random_state(estimator)
    assert estimator.fit(X, y) is estimator


def check_f_contiguous_array_estimator(name, estimator_orig):
    # Non-regression test for F-contiguous input handling.
    estimator = clone(estimator_orig)

    X, y = _binary_Xy(n_samples=20)
    X = _enforce_estimator_tags_X(estimator_orig, X)
    X = np.asfortranarray(X)
    y = _enforce_estimator_tags_y(estimator_orig, y)

    estimator.fit(X, y)

    if hasattr(estimator, "transform"):
        estimator.transform(X)
    if hasattr(estimator, "predict"):
        estimator.predict(X)  # pragma: no cover


def check_transformer_general(name, transformer_orig):
    if not hasattr(transformer_orig, "transform"):
        return
    X, y = _binary_Xy(n_samples=30)
    X = _enforce_estimator_tags_X(transformer_orig, X)
    _check_transformer_binary(name, transformer_orig, X, y)


def _check_transformer_binary(name, transformer_orig, X, y):
    # The bool-compatible core of sklearn's _check_transformer: fit_transform
    # equals fit().transform(), shapes are consistent, and transform rejects a
    # wrong feature count. The dtype-preservation parts are omitted (they need
    # float input -- see check_transformer_preserve_dtypes).
    n_samples, n_features = np.asarray(X).shape
    transformer = clone(transformer_orig)
    set_random_state(transformer)

    transformer.fit(X, y)
    # fit_transform should work on a non-fitted estimator
    transformer_clone = clone(transformer)
    X_pred = transformer_clone.fit_transform(X, y=y)
    assert X_pred.shape[0] == n_samples

    if hasattr(transformer, "transform"):
        X_pred2 = transformer.transform(X)
        X_pred3 = transformer.fit_transform(X, y=y)

        if get_tags(transformer_orig).non_deterministic:
            pytest.skip(name + " is non deterministic")  # pragma: no cover

        assert_allclose_dense_sparse(
            X_pred,
            X_pred2,
            atol=1e-2,
            err_msg=f"fit_transform and transform outcomes not consistent in {name}",
        )
        assert_allclose_dense_sparse(
            X_pred,
            X_pred3,
            atol=1e-2,
            err_msg=f"consecutive fit_transform outcomes not consistent in {name}",
        )
        assert X_pred2.shape[0] == n_samples
        assert X_pred3.shape[0] == n_samples

        if (
            hasattr(X, "shape")
            and get_tags(transformer).requires_fit
            and X.ndim == 2
            and X.shape[1] > 1
        ):
            with pytest.raises(ValueError):
                transformer.transform(X[:, :-1])


def check_estimators_nan_inf(name, estimator_orig):
    # Checks that the estimator rejects NaN/inf. A bool array cannot hold NaN or
    # inf, so -- like sklearn -- this necessarily uses non-finite *float* input;
    # scihfs's validate_data finiteness check fires before the bool-dtype gate,
    # so the rejection is genuinely about non-finiteness (message matches
    # inf/NaN). The finite baseline is fitted on bool (float would be rejected
    # for dtype before its finiteness ever mattered).
    X_train_finite, _ = _binary_Xy(n_samples=10)
    rnd = np.random.RandomState(0)
    X_train_nan = rnd.uniform(size=(10, _N_FEATURES))
    X_train_nan[0, 0] = np.nan
    X_train_inf = rnd.uniform(size=(10, _N_FEATURES))
    X_train_inf[0, 0] = np.inf
    y = np.ones(10)
    y[:5] = 0
    y = _enforce_estimator_tags_y(estimator_orig, y)

    for X_train in [X_train_nan, X_train_inf]:
        estimator = clone(estimator_orig)
        set_random_state(estimator, 1)
        # fit must reject the non-finite input
        with pytest.raises(ValueError, match=r"inf|NaN"):
            estimator.fit(X_train, y)
        # fit on finite (bool) data, then check predict/transform reject it too
        estimator.fit(X_train_finite, y)
        if hasattr(estimator, "predict"):  # pragma: no cover
            with pytest.raises(ValueError, match=r"inf|NaN"):
                estimator.predict(X_train)
        if hasattr(estimator, "transform"):
            with pytest.raises(ValueError, match=r"inf|NaN"):
                estimator.transform(X_train)


def check_transformer_preserve_dtypes(name, transformer_orig):
    # Check that dtype is preserved: bool in -> bool out. scihfs accepts only
    # bool, so bool is the single supported dtype to preserve (sklearn iterates
    # transformer_tags.preserves_dtype, which lists float dtypes the contract
    # rejects). Classifier-only estimators have no transform.
    if not hasattr(transformer_orig, "transform"):
        return
    transformer = clone(transformer_orig)
    X, y = _binary_Xy(n_samples=30)
    X = _enforce_estimator_tags_X(transformer_orig, X)

    set_random_state(transformer)
    X_trans1 = transformer.fit_transform(X, y)
    X_trans2 = transformer.fit(X, y).transform(X)

    for Xt, method in zip([X_trans1, X_trans2], ["fit_transform", "transform"]):
        assert Xt.dtype == X.dtype, (
            f"{name} (method={method}) does not preserve dtype. "
            f"Original/Expected dtype={X.dtype}, got dtype={Xt.dtype}."
        )


_SECTION_B_CHECKS = [
    check_estimator_sparse_array,
    check_estimator_sparse_matrix,
    check_estimator_sparse_tag,
    check_readonly_memmap_input,
    check_f_contiguous_array_estimator,
    check_transformer_general,
    check_estimators_nan_inf,
    check_transformer_preserve_dtypes,
]


# ---------------------------------------------------------------------------
# Section 2C:
# Stubs ('pass') for those sklearn checks that fundamentally require
# different data and cannot be faithfully recreated.
# ---------------------------------------------------------------------------


def check_dtype_object(name, estimator_orig):
    # STUB: sklearn feeds object-dtype X and expects it treated as numeric. The
    # bool-dtype contract rejects any non-bool dtype (object included), so the
    # "treat object as numeric" intent cannot be exercised.
    pass


def check_estimators_dtypes(name, estimator_orig):
    # STUB: sklearn fits on float32/float64/int X variants and expects all to
    # work. All are non-bool and rejected by the contract.
    pass


def check_positive_only_tag_during_fit(name, estimator_orig):
    # STUB: probes the positive_only tag by feeding negative values. A bool
    # array has no negative values, and non-bool input is rejected for dtype
    # (not for sign), so the tag cannot be exercised here.
    pass


_SECTION_C_CHECKS = [
    check_dtype_object,
    check_estimators_dtypes,
    check_positive_only_tag_during_fit,
]

_SKLEARN_ADJACENT_CHECKS = _SECTION_A_CHECKS + _SECTION_B_CHECKS + _SECTION_C_CHECKS


@pytest.mark.parametrize("check", _SKLEARN_ADJACENT_CHECKS, ids=lambda c: c.__name__)
@pytest.mark.parametrize("estimator", ALL_ESTIMATORS)
def test_binary_conformance(estimator, check):
    est = estimator(_HIERARCHY)
    check(type(est).__name__, est)
