import math
import random

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from info_gain.info_gain import info_gain, info_gain_ratio

from hfs.helpers import get_columns_for_numpy_hierarchy
from hfs.metrics import cosine_similarity


@pytest.fixture()
def data1():
    X = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    hierarchy = hierarchy1()
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data1_2():
    X = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    hierarchy = hierarchy1_2()
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data2():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    hierarchy = hierarchy2()
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data2_1():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    edges = [(0, 1), (1, 2), (1, 3)]
    hierarchy = nx.DiGraph(edges)
    hierarchy.add_node(4)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data2_2():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    edges = [(0, 1), (1, 2), (1, 3)]
    hierarchy = nx.DiGraph(edges)
    hierarchy.add_node(4)
    hierarchy.add_edge(4, 3)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data3():
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )

    hierarchy = hierarchy3()
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data4():
    X = np.array(
        [
            [1, 1, 0, 1, 1, 1, 1],
            [1, 1, 1, 0, 1, 0, 0],
            [1, 1, 1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0, 1, 0],
            [1, 1, 0, 0, 1, 1, 1],
        ],
    )
    edges = [(0, 1), (1, 2), (0, 3), (0, 4), (0, 5), (5, 6)]
    hierarchy = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([1, 0, 0, 1, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data_numerical():
    X = np.array(
        [
            [1, 6, 3, 0, 1],
            [4, 7, 1, 7, 0],
            [2, 2, 5, 4, 0],
            [6, 0, 2, 0, 2],
            [1, 4, 1, 0, 3],
        ]
    )
    hierarchy = hierarchy1()
    columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def hierarchy1():
    edges = [(0, 1), (1, 2), (0, 3), (0, 4)]
    return nx.DiGraph(edges)


@pytest.fixture()
def hierarchy1_2():
    edges = [(0, 4), (0, 3), (0, 1), (1, 2)]
    return nx.DiGraph(edges)


@pytest.fixture()
def hierarchy2():
    edges = [(0, 1), (1, 2), (2, 3), (0, 4)]
    return nx.DiGraph(edges)


@pytest.fixture()
def hierarchy3():
    hierarchy = nx.DiGraph()
    hierarchy.add_nodes_from([0, 1, 2, 3, 4])
    return hierarchy


@pytest.fixture()
def dataframe():
    return pd.DataFrame(
        {
            4: [4, 4, 4, 4, 4],
            2: [2, 2, 2, 2, 2],
            0: [0, 0, 0, 0, 0],
            1: [1, 1, 1, 1, 1],
            3: [3, 3, 3, 3, 3],
        }
    )


@pytest.fixture()
def result_tsel1():
    result = np.array([[0], [0], [0], [0], [1]])
    support = np.array([True, False, False, False, False])
    return (result, support)


@pytest.fixture()
def result_tsel2():
    result = np.array(
        [
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [1, 0],
        ]
    )
    support = np.array([False, True, False, False, True])
    return (result, support)


@pytest.fixture()
def result_tsel3():
    result = data3()[0]
    support = np.array([True, True, True, True, True])
    return (result, support)


@pytest.fixture()
def result_shsel1():
    return result_tsel1()


@pytest.fixture()
def result_shsel_hfe1():
    return result_shsel1()


@pytest.fixture()
def result_shsel2():
    result = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    support = np.array([False, False, True, False, True])
    return (result, support)


@pytest.fixture()
def result_shsel_hfe2():
    result = np.array(
        [
            [0],
            [1],
            [1],
            [0],
            [0],
        ],
    )
    support = np.array([False, False, True, False, False])
    return (result, support)


@pytest.fixture()
def result_shsel_hfe4():
    result = np.array(
        [
            [0, 1, 1],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 1, 1],
        ],
    )
    support = np.array([False, False, True, False, False, True, True])
    return (result, support)


@pytest.fixture()
def result_shsel3():
    return result_tsel3()


@pytest.fixture()
def data_shsel_selection():
    X = data2()[0]
    y = data2()[1]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    columns = None
    return (X, y, hierarchy, columns)


@pytest.fixture()
def result_shsel_selection():
    result = np.array(
        [
            [0],
            [1],
            [1],
            [0],
            [0],
        ],
    )
    support = np.array([False, False, True, False, False])
    return (result, support)


@pytest.fixture()
def result_gtd_selection2():
    result = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    support = np.array([False, False, True, False, True])
    return (result, support)


@pytest.fixture()
def result_gtd_selection2_1():
    result = np.array(
        [
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
        ],
    )
    support = np.array([False, False, True, True, True])
    return (result, support)


@pytest.fixture()
def result_gtd_selection2_2():
    result = np.array(
        [
            [0, 1],
            [1, 0],
            [1, 0],
            [0, 1],
            [0, 0],
        ],
    )
    support = np.array([False, False, True, False, True])
    return (result, support)


@pytest.fixture()
def result_hill_selection_td():
    result = pd.DataFrame(
        [
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ],
    )
    support = np.array([False, True, False, True, True])
    return (result, support)


@pytest.fixture()
def result_hill_selection_bu():
    k = 3
    result = np.array(
        [
            [0, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    support = np.array([False, False, True, True, True])
    return (result, support, k)


@pytest.fixture()
def wrong_hierarchy_X():
    X = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1)]))
    columns = [0, 1]
    return (X, hierarchy, columns)


@pytest.fixture()
def wrong_hierarchy_X1():
    X = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    hierarchy = nx.to_numpy_array(nx.DiGraph([(0, 1), (1, 2), (3, 4), (0, 5)]))
    columns = [0, 1, 2]
    return (X, hierarchy, columns)


@pytest.fixture()
def result_score_matrix1():
    return np.array(
        [
            [1, 0, 0, 0, 1],
            [2, 0, 0, 1, 1],
            [3, 1, 1, 1, 1],
            [4, 2, 1, 1, 1],
            [5, 2, 1, 1, 1],
        ]
    )


@pytest.fixture()
def result_score_matrix_numerical():
    return np.array(
        [
            [
                1.6931471805599454,
                1.5978370007556206,
                1.241162056816888,
                0,
                1.0870113769896297,
            ],
            [
                1.6931471805599454,
                1.3513978868378886,
                1.0512932943875506,
                1.3136575588550417,
                0,
            ],
            [
                1.6931471805599454,
                1.4307829160924541,
                1.325422400434628,
                1.2682639865946794,
                0,
            ],
            [
                1.6931471805599454,
                1.1823215567939547,
                1.1823215567939547,
                0,
                1.1823215567939547,
            ],
            [
                1.6931471805599454,
                1.4418327522790393,
                1.1053605156578263,
                0,
                1.2876820724517808,
            ],
        ]
    )


@pytest.fixture()
def result_comparison_matrix_td1():
    return np.array(
        [
            [0.0, math.sqrt(2), math.sqrt(7), math.sqrt(15), math.sqrt(22)],
            [math.sqrt(2), 0.0, math.sqrt(3), math.sqrt(9), math.sqrt(14)],
            [math.sqrt(7), math.sqrt(3), 0.0, math.sqrt(2), math.sqrt(5)],
            [math.sqrt(15), math.sqrt(9), math.sqrt(2), 0.0, 1.0],
            [math.sqrt(22), math.sqrt(14), math.sqrt(5), 1.0, 0.0],
        ]
    )


@pytest.fixture()
def result_comparison_matrix_bu(matrix: np.ndarray):
    result = np.zeros((5, 5))
    for x in range(5):
        for y in range(5):
            result[x, y] = cosine_similarity(matrix[x, :], matrix[y, :])
    return result


@pytest.fixture()
def result_comparison_matrix_bu1():
    matrix = result_score_matrix1()
    return result_comparison_matrix_bu(matrix)


@pytest.fixture()
def result_fitness_funtion_td1():
    alpha = 0.99
    doc1 = math.sqrt(22) / (1 + alpha * (math.sqrt(2) + math.sqrt(7) + math.sqrt(15)))
    doc2 = math.sqrt(14) / (1 + alpha * (math.sqrt(2) + math.sqrt(3) + math.sqrt(9)))
    doc3 = math.sqrt(5) / (1 + alpha * (math.sqrt(7) + math.sqrt(3) + math.sqrt(2)))
    doc4 = 1.0 / (1 + alpha * (math.sqrt(15) + math.sqrt(9) + math.sqrt(2)))
    doc5 = (math.sqrt(22) + math.sqrt(14) + math.sqrt(5) + 1.0) / 1.0

    return doc1 + doc2 + doc3 + doc4 + doc5


@pytest.fixture()
def result_fitness_funtion_bu1():
    alpha = 3
    n = 5
    beta = 0.01
    k = 3
    selected_nearest_neighbors = [[1, 2], [2, 0], [3, 1], [1, 2], []]
    result = sum([len(x) for x in selected_nearest_neighbors])
    result = result * (1 + beta * (alpha - n) / alpha)
    return (result, k)


@pytest.fixture()
def result_score_matrix2():
    return np.array(
        [
            [3, 1, 0, 0, 1],
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0, 0],
            [2, 0, 0, 0, 1],
            [2, 1, 0, 0, 0],
        ]
    )


@pytest.fixture()
def result_comparison_matrix_bu2():
    matrix = result_score_matrix2()
    return result_comparison_matrix_bu(matrix)


@pytest.fixture()
def result_score_matrix3():
    return np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )


@pytest.fixture()
def result_comparison_matrix_bu3():
    matrix = result_score_matrix3()
    return result_comparison_matrix_bu(matrix)


@pytest.fixture()
def result_gr_values2():
    y = np.array([1, 0, 0, 1, 1])
    return [
        info_gain_ratio(np.array([1, 1, 1, 1, 1]), y),
        info_gain_ratio(np.array([1, 1, 1, 0, 1]), y),
        info_gain_ratio(np.array([0, 1, 1, 0, 0]), y),
        info_gain_ratio(np.array([0, 1, 0, 0, 0]), y),
        info_gain_ratio(np.array([1, 0, 0, 1, 0]), y),
    ]


@pytest.fixture()
def result_ig_values2():
    y = np.array([1, 0, 0, 1, 1])
    return [
        info_gain(np.array([1, 1, 1, 1, 1]), y),
        info_gain(np.array([1, 1, 1, 0, 1]), y),
        info_gain(np.array([0, 1, 1, 0, 0]), y),
        info_gain(np.array([0, 1, 0, 0, 0]), y),
        info_gain(np.array([1, 0, 0, 1, 0]), y),
    ]


@pytest.fixture()
def result_aggregated1():
    return np.array(
        [
            [1, 0, 0, 0, 1],
            [2, 0, 0, 1, 1],
            [3, 1, 1, 1, 1],
            [4, 2, 1, 1, 1],
            [5, 2, 1, 1, 1],
        ]
    )


@pytest.fixture()
def result_aggregated2():
    return np.array(
        [
            [3, 1, 0, 0, 1],
            [4, 3, 2, 1, 0],
            [3, 2, 1, 0, 0],
            [2, 0, 0, 0, 1],
            [2, 1, 0, 0, 0],
        ],
    )


# TODO should this be capitalised if its a constant?
_feature_number = 9


# TODO is this a fixture or a method? if method move to helpers?
def getFixedDag():
    return nx.to_numpy_array(
        nx.DiGraph(
            [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (4, 6), (4, 7), (3, 7), (5, 8)]
        )
    )


# TODO what is this rand about? where is it used?
def rand():
    return random.getrandbits(1)


def random_lines_with_assertions(y):
    b = rand()
    c = rand()
    d = rand()
    e = rand() if b == 1 else 0
    f = rand() if b == 1 else 0
    g = rand() if e * d == 1 else 0
    h = rand() if e == 1 and d == 1 else 0
    i = rand() if f == 1 else 0
    return (1, b, c, d, e, f, g, h, i)


# TODO not used anywhere?
def get_fixed_data(instance_number):
    df = pd.DataFrame(columns=[i for i in range(0, _feature_number)])
    y = np.random.randint(0, 2, instance_number)
    for row in range(0, instance_number):
        df.loc[len(df)] = random_lines_with_assertions(y)
    return df.to_numpy(), y


@pytest.fixture()
def lazy_data1():
    edges = [
        (9, 3),
        (9, 7),
        (7, 1),
        (3, 1),
        (7, 6),
        (1, 6),
        (1, 5),
        (6, 8),
        (3, 0),
        (4, 0),
        (1, 5),
        (2, 0),
        (10, 2),
        (4, 11),
        (5, 11),
    ]
    hierarchy = nx.DiGraph(edges)
    X_train = np.ones((2, len(hierarchy.nodes)))
    y_train = np.array([0, 1])
    X_test = np.array(
        [[1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0], [1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0]]
    )
    y_test = np.array([1, 0])
    relevance = [0.25, 0.23, 0.38, 0.25, 0.28, 0.38, 0.26, 0.31, 0.26, 0.23, 0.21, 0.26]

    return (
        hierarchy,
        X_train,
        y_train,
        X_test,
        y_test,
        relevance,
    )


@pytest.fixture()
def lazy_data2():
    small_DAG = nx.to_numpy_array(nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3)]))
    train_x_data = np.array([[1, 1, 0, 1], [1, 0, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]])
    train_y_data = np.array([0, 0, 1, 1])
    test_x_data = np.array([[1, 1, 0, 0], [1, 1, 1, 0]])
    test_y_data = np.array([1, 0])
    return (small_DAG, train_x_data, train_y_data, test_x_data, test_y_data)


@pytest.fixture()
def lazy_data3():
    edges = [(4, 0), (0, 3), (2, 3), (5, 2), (5, 1)]
    hierarchy = nx.DiGraph(edges)
    X_train_ones = np.ones((9, len(hierarchy.nodes)))
    X_train = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 1, 0, 1],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 0, 1],
        ]
    )
    y_train = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1])
    X_test = np.array([[0, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]])
    y_test = np.array([1, 0])
    resulted_features = np.array(
        [[0.0, 1.0, 1.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]]
    )
    return (
        hierarchy,
        X_train_ones,
        X_train,
        y_train,
        X_test,
        y_test,
        resulted_features,
    )


@pytest.fixture()
def lazy_data4():
    big_DAG = getFixedDag()
    small_DAG = nx.to_numpy_array(nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3)]))
    return small_DAG, big_DAG
