import numpy as np
import pytest
import networkx as nx
from sklearn.datasets import make_classification

from hfs.selectors import GASel
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


@pytest.mark.parametrize("data", ["zero_sensitivity_case", "zero_specificity_case"])
def test_geometric_mean_sensitivity_specificity(data, request):
    y_true, y_pred = request.getfixturevalue(data)
    score = geometric_mean_sensitivity_specificity(y_true, y_pred)
    assert isinstance(score, float)


@pytest.mark.parametrize("data", ["zero_sensitivity_case", "zero_specificity_case"])
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


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=20, random_state=42)
    return X, y


@pytest.fixture
def hierarchy():
    graph = nx.DiGraph([(0, 2), (0, 3), (1, 4), (1, 5), (2, 6), (3, 7), (4, 8), (5, 9)])
    adj = nx.to_numpy_array(graph, nodelist=[i for i in range(10)])

    return adj


@pytest.fixture
def ga_selector(hierarchy):
    return GASel(hierarchy=hierarchy, n_population=10, n_generations=5)


def test_initialize_population(ga_selector, sample_data):
    X, y = sample_data
    population = ga_selector._initialize_population(X.shape[1])
    assert population.shape == (ga_selector.n_population, X.shape[1])
    assert np.all((population == 0) | (population == 1)), "Population must be binary."


def test_fitness(ga_selector, sample_data):
    X, y = sample_data
    individual = np.random.randint(2, size=X.shape[1])
    fitness = ga_selector._fitness(individual, X, y)
    assert isinstance(fitness, float), "Fitness must be a float value."


def test_selection(ga_selector, sample_data):
    X, y = sample_data
    population = ga_selector._initialize_population(X.shape[1])
    fitness_scores = np.array([ga_selector._fitness(ind, X, y) for ind in population])
    parents = ga_selector._selection(fitness_scores)
    assert len(parents) == 2, "Two parents should be selected."


def test_mutation(ga_selector):
    individual = np.random.randint(2, size=20)
    mutated_individual = ga_selector._mutation(individual)
    assert len(mutated_individual) == len(
        individual
    ), "Mutated individual should have the same length."


def test_fit_transform(ga_selector, sample_data):
    X, y = sample_data
    ga_selector.fit(X, y)
    transformed_X = ga_selector.transform(X)
    assert (
        transformed_X.shape[1] <= X.shape[1]
    ), "Transformed X should have fewer or equal features."


def test_ancestors(hierarchy, ga_selector):
    assert ga_selector.ancestors(6) == {
        0,
        2,
    }, "Ancestors of feature 6 should be {0, 'ROOT'}."


def test_has_ancestors(ga_selector):
    assert ga_selector.has_ancestors(6), "Feature 6 should have ancestors."
    assert not ga_selector.has_ancestors(0), "Feature 0 should not have ancestors."


def test_descendants(hierarchy, ga_selector):
    assert ga_selector.descendants(1) == {
        4,
        5,
        8,
        9,
    }, "Descendants of feature 0 should be {4, 5, 8, 9}."


def test_has_descendants(ga_selector):
    assert ga_selector.has_descendants(0), "Feature 0 should have descendants."
    assert not ga_selector.has_descendants(7), "Feature 7 should not have descendants."