import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from hfs.selectors import HierarchicalEstimator  # assuming the base class is imported
from sklearn.naive_bayes import BernoulliNB
def _crossover(parent1, parent2):
    """
    Crossover two parents and get two children.

    Parameters
    __________
    parent1 : np.ndarray or list
        The first parent of the crossover.
    parent2 : np.ndarray or list
        The second parent of the crossover.

    Returns
    __________
    child1 : np.ndarray or list
        The first child of the crossover.
    child2 : np.ndarray or list
        The second child of the crossover.
    """
    point = np.random.randint(len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def geometric_mean_sensitivity_specificity(y_true, y_pred):
    """
    Compute the geometric mean of sensitivity and specificity scores.

    Parameters
    __________
    y_true : array-like
        The true labels.
    y_pred : array-like
        Predicted labels.

    Returns
    __________
    gmss: float
        The geometric mean of sensitivity and specificity scores.

    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)

    # Division by zero check
    denominator = conf_matrix[0, 0] + conf_matrix[0, 1]
    if denominator == 0:
        # If TN, FP = 0 specificity is not defined
        specificity = 0
    else:
        specificity = conf_matrix[0, 0] / denominator

    return np.sqrt(sensitivity * specificity)

class GASel(HierarchicalEstimator):
    """Genetic Algorithm Based Feature Selector

    Implements approach from da Silva et al.

    """
    def __init__(self,
                 hierarchy=None,
                 n_population=50,
                 n_generations=50,
                 mutation_prob=0.02,
                 she_mutation_prob=0.3,
                 epsilon=0.05
                 ):
        """Initialize the GA selector.

        Parameters
        __________
        hierarchy : np.ndarray
                    The hierarchy of the model in its adjacency matrix representation.
        n_population : int
                    The population size.
        mutation_prob : float
                    The mutation probability.
        she_mutation_prob : float
                     The Simple Hierarchical Elimination mutation probability.
        epsilon : float
                     Epsilon parameter of genetic algorithm elitism.
        """
        super().__init__(hierarchy=hierarchy)
        # Hierarchy setting in both ways: nx.Digraph and adj matrix
        super()._set_hierarchy()
        self.estimator = BernoulliNB()  # Assume a default estimator if None provided
        self.cv = cv
        self.n_population = n_population
        self.n_generations = n_generations
        self.mutation_prob = mutation_prob
        self.she_mutation_prob = she_mutation_prob
        self.epsilon = epsilon
        self.selected_features_ = None

    def _initialize_population(self, n_features):
        return np.random.randint(2, size=(self.n_population, n_features))

    def _fitness(self, individual, X, y):
        if individual.sum() == 0:
            return 0
        X_selected = X[:, individual == 1]
        gm_scorer = make_scorer(geometric_mean_sensitivity_specificity)
        scores = cross_val_score(self.estimator, X_selected, y, cv=StratifiedKFold(self.cv), scoring=gm_scorer)
        penalty_for_features = 0.01 * X_selected.shape[1] / X.shape[1]
        return scores.mean() - penalty_for_features

    def _selection(self, fitness_scores):
        return np.random.choice(len(fitness_scores), 2, replace=False)

    def _mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        return individual

    def _fit(self, X, y):
        n_features = X.shape[1]
        population = self._initialize_population(n_features)
        for _ in range(self.n_generations):
            fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])
            best_indices = fitness_scores >= np.quantile(fitness_scores, 1 - self.epsilon)
            new_population = population[best_indices].tolist()
            while len(new_population) < self.n_population:
                parent_indices = self._selection(fitness_scores)
                parent1, parent2 = population[parent_indices[0]], population[parent_indices[1]]
                children = _crossover(parent1, parent2)
                new_population.extend([self._mutation(child) for child in children])
            population = np.array(new_population[:self.n_population])
        self.selected_features_ = population[np.argmax(fitness_scores)]

    def fit(self, X, y=None, columns=None):
        #super().fit(X, y, columns)
        self._columns = columns if columns is not None else list(range(X.shape[1]))
        self._fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X):
        if not hasattr(self, 'selected_features_') or self.selected_features_ is None:
            raise ValueError("The model has not been fitted yet.")
        return X[:, self.selected_features_ == 1]

    def has_successors(self, feature_index):
        node = self.get_columns()[feature_index]
        return any(True for _ in self._hierarchy.successors(node))
