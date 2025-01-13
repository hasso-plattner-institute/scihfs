import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from hfs.helpers import get_columns_for_numpy_hierarchy
from hfs.metrics import cosine_similarity


@pytest.fixture()
def data1(hierarchy1):
    X = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    columns = get_columns_for_numpy_hierarchy(hierarchy1, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy1)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data1_2(hierarchy1_2):
    X = np.array(
        [
            [0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 1, 1, 1],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    columns = get_columns_for_numpy_hierarchy(hierarchy1_2, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy1_2)
    y = np.array([0, 0, 0, 0, 1])
    return (X, y, hierarchy, columns)


@pytest.fixture()
def data2(hierarchy2):
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )
    columns = get_columns_for_numpy_hierarchy(hierarchy2, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy2)
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
def data3(hierarchy3):
    X = np.array(
        [
            [1, 1, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0],
        ],
    )

    columns = get_columns_for_numpy_hierarchy(hierarchy3, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy3)
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
def data_numerical(hierarchy1):
    X = np.array(
        [
            [1, 6, 3, 0, 1],
            [4, 7, 1, 7, 0],
            [2, 2, 5, 4, 0],
            [6, 0, 2, 0, 2],
            [1, 4, 1, 0, 3],
        ]
    )
    columns = get_columns_for_numpy_hierarchy(hierarchy1, X.shape[1])
    hierarchy = nx.to_numpy_array(hierarchy1)
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
def result_tsel3(data3):
    result = data3[0]
    support = np.array([True, True, True, True, True])
    return (result, support)


@pytest.fixture()
def result_shsel1(result_tsel1):
    return result_tsel1


@pytest.fixture()
def result_shsel_hfe1(result_shsel1):
    return result_shsel1


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
def result_shsel3(result_tsel3):
    return result_tsel3


@pytest.fixture()
def data_shsel_selection(data2):
    X = data2[0]
    y = data2[1]
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


# TODO maybe move to helper
def result_comparison_matrix_bu(matrix: np.ndarray):
    result = np.zeros((5, 5))
    for x in range(5):
        for y in range(5):
            result[x, y] = cosine_similarity(matrix[x, :], matrix[y, :])
    return result


@pytest.fixture()
def result_comparison_matrix_bu1(result_score_matrix1):
    return result_comparison_matrix_bu(result_score_matrix1)


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
def result_comparison_matrix_bu2(result_score_matrix2):
    return result_comparison_matrix_bu(result_score_matrix2)


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
def result_comparison_matrix_bu3(result_score_matrix3):
    return result_comparison_matrix_bu(result_score_matrix3)


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


# TODO is this a fixture or a method? if method move to helpers?
def get_fixed_dag():
    return nx.to_numpy_array(
        nx.DiGraph(
            [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (4, 6), (4, 7), (3, 7), (5, 8)]
        )
    )


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
    big_DAG = get_fixed_dag()
    small_DAG = nx.to_numpy_array(nx.DiGraph([(0, 1), (0, 2), (1, 2), (1, 3)]))
    return small_DAG, big_DAG


@pytest.fixture()
def data1_preprocessing():
    X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    edges = [(1, 3), (3, 2), (0, 4), (0, 1)]
    hierarchy_original = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy_original, X.shape[1])
    hierarchy_original = nx.to_numpy_array(hierarchy_original)
    hierarchy_transformed = nx.to_numpy_array(nx.DiGraph([(1, 3), (3, 2), (0, 1)]))
    X_transformed = np.array([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]])

    return (X, X_transformed, hierarchy_original, columns, hierarchy_transformed)


@pytest.fixture()
def data2_preprocessing():
    X = np.array([[0, 0, 1], [0, 1, 0], [0, 1, 0]])

    edges = [(1, 2), (1, 3), (3, 4), (0, 1)]
    hierarchy_original = nx.DiGraph(edges)
    columns = get_columns_for_numpy_hierarchy(hierarchy_original, X.shape[1])
    hierarchy_original = nx.to_numpy_array(hierarchy_original)
    edges_tranformed = [(1, 2), (0, 1)]
    hierarchy_transformed = nx.to_numpy_array(nx.DiGraph(edges_tranformed))
    X_transformed = np.array([[1, 1, 1], [1, 1, 0], [1, 1, 0]])

    return (X, X_transformed, hierarchy_original, columns, hierarchy_transformed)


# currently only used for test_shrink_dag
@pytest.fixture()
def data3_preprocessing():
    edges = [
        ("GO:2001090", "GO:2001091"),
        ("GO:2001090", "GO:2001092"),
        ("GO:2001091", "GO:2001093"),
        ("GO:2001091", "GO:2001094"),
        ("GO:2001093", "GO:2001095"),
    ]
    #      0
    #   1      2
    # 3    4
    # 5
    hierarchy = nx.to_numpy_array(nx.DiGraph(edges))
    X_identifiers = list([0, 1, 2, 4])
    X = np.ones((2, len(X_identifiers)))
    # in X there is 0,1,2,4
    edges_transformed = [
        ("GO:2001090", "GO:2001091"),
        ("GO:2001090", "GO:2001092"),
        ("GO:2001091", "GO:2001094"),
    ]
    h = nx.DiGraph(edges_transformed)

    h.add_edge("ROOT", "GO:2001090")

    hierarchy_transformed = nx.to_numpy_array(h)

    return (X, hierarchy, hierarchy_transformed, X_identifiers)
