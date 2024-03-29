import numpy as np
import pytest

from hfs.selectors import HieAODE, HieAODE_plus

from .fixtures.fixtures import *

SMOOTHING = 1
PRIOR = 0.5


@pytest.fixture
def expected_prior_lazy_data2():
    return np.array(
        [
            [  # feature 0
                [  # Class 0
                    0.0,  # Value 0
                    0.5,  # Value 1
                ],
                [  # Class 1
                    0.0,  # Value 0
                    0.5,  # Value 1
                ],
            ],
            [  # feature 1
                [  # Class 0
                    0.25,  # Value 0
                    0.25,  # Value 1
                ],
                [  # Class 1
                    0.0,  # Value 0
                    0.5,  # Value 1
                ],
            ],
            [  # feature 2
                [  # Class 0
                    0.5,  # Value 0
                    0.0,  # Value 1
                ],
                [  # Class 1
                    0.0,  # Value 0
                    0.5,  # Value 1
                ],
            ],
            [  # feature 3
                [  # Class 0
                    0.25,  # Value 0
                    0.25,  # Value 1
                ],
                [  # Class 1
                    0.25,  # Value 0
                    0.25,  # Value 1
                ],
            ],
        ]
    )


@pytest.fixture
def expected_ancestors_lazy_data2():
    ancestors = np.full((4, 2, 2), -1, dtype=float)  # ancestor idx, class, value
    ancestors[0] = [  # feature 0 as ancestor
        [  # Class 0
            (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 0
            (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 1
        ],
        [  # Class 1
            (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 0
            (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 1
        ],
    ]
    ancestors[1] = [  # feature 1 as ancestor
        [  # Class 0
            (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 0
            (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 1
        ],
        [  # Class 1
            (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 0
            (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Value 1
        ],
    ]
    return ancestors


@pytest.fixture
def expected_descendants_lazy_data2():
    descendants = np.full(
        (4, 4, 2, 2, 2), -1, dtype=float
    )  # descendant idx, feature idx, class, descendant value, feature value
    descendants[1][0] = [  # feature 1 is a descendant of feature 0
        [  # Class 0
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    descendants[2][0] = [  # feature 2 is a descendant of feature 0
        [  # Class 0
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    descendants[3][0] = [  # feature 3 is a descendant of feature 0
        [  # Class 0
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    descendants[2][1] = [  # feature 2 is a descendant of feature 1
        [  # Class 0
            [  # Descendant value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (2.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    descendants[3][1] = [  # feature 3 is a descendant of feature 1
        [  # Class 0
            [  # Descendant value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    descendants[3][2] = [  # feature 3 is a descendant of feature 2
        [  # Class 0
            [  # Descendant value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (0.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (2.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    descendants[2][3] = [  # feature 2 is a descendant of feature 3
        [  # Class 0
            [  # Descendant value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
        ],
        [  # Class 1
            [  # Descendant value 0
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (0.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
            [  # Descendant value 1
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 0
                (1.0 + SMOOTHING * PRIOR) / (1.0 + SMOOTHING),  # Feature value 1
            ],
        ],
    ]
    return descendants


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_calculate_dependency_ascendant_class(
    data,
    expected_prior_lazy_data2,
    expected_ancestors_lazy_data2,
    expected_descendants_lazy_data2,
):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    selector = HieAODE(hierarchy=small_DAG)
    selector.fit_selector(
        X_train=train_x_data, y_train=train_y_data, X_test=test_x_data
    )

    pred = selector.select_and_predict(predict=True, saveFeatures=True)
    result_prior_lazy_data2 = selector.cpts["prior"]
    result_ancestors_lazydata2 = selector.cpts["ancestors"]
    result_descendants_lazydata2 = selector.cpts["descendants"]
    assert assert_arrays_equal(
        result_prior_lazy_data2, expected_prior_lazy_data2, "prior"
    )
    assert assert_arrays_equal(
        result_ancestors_lazydata2, expected_ancestors_lazy_data2, "ancestors"
    )
    assert assert_arrays_equal(
        result_descendants_lazydata2, expected_descendants_lazy_data2, "descendants"
    )


HEADERS = dict(
    prior=["Feature idx", "Class", "Value", "Actual", "Expected"],
    ancestors=["Ancestor idx", "Class", "Value", "Actual", "Expected"],
    descendants=[
        "Descendant idx",
        "Feature idx",
        "Class",
        "DescendantV",
        "FeatureV",
        "Actual",
        "Expected",
    ],
)


def assert_arrays_equal(actual, expected, header_name):
    if not np.array_equal(actual, expected):
        assert actual.shape == expected.shape, "Arrays must have the same shape"
        # Find indices where elements are different
        indices = np.where(actual != expected)

        # Create str table
        current_headers = HEADERS[header_name]
        data = {current_headers[i]: indices[i] for i in range(actual.ndim)}
        data["Actual"] = actual[indices]
        data["Expected"] = expected[indices]
        df = pd.DataFrame(data)
        table_str = df.to_markdown(index=False)

        raise AssertionError("\n" + table_str)
    else:
        return True


@pytest.mark.parametrize(
    "data",
    [
        lazy_data2(),
    ],
)
def test_hieaode_plus_plus(data):
    small_DAG, train_x_data, train_y_data, test_x_data, test_y_data = data
    selector = HieAODE_plus(hierarchy=small_DAG)
    selector.fit_selector(
        X_train=train_x_data, y_train=train_y_data, X_test=test_x_data
    )

    pred = selector.select_and_predict(predict=True, saveFeatures=True)