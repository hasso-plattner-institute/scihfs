import networkx as nx
import numpy as np
from sklearn.utils.validation import check_X_y
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from hfs.selectors import HierarchicalEstimator  # assuming the base class is imported
from sklearn.naive_bayes import BernoulliNB
from scipy.stats.stats import pearsonr


def _crossover(parent1, parent2):
    """
    Perform a crossover between two parent individuals to produce two offspring.

    Parameters
    ----------
    parent1 : np.ndarray
        The first parent individual's feature presence array.
    parent2 : np.ndarray
        The second parent individual's feature presence array.

    Returns
    -------
    tuple
        A tuple containing two np.ndarrays representing the offspring.
    """
    point = np.random.randint(len(parent1))
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def geometric_mean_sensitivity_specificity(y_true, y_pred):
    """
    Calculate the geometric mean of sensitivity and specificity.

    Parameters
    ----------
    y_true : array-like
        True class labels.
    y_pred : array-like
        Predicted class labels by the model.

    Returns
    -------
    float
        The geometric mean of sensitivity and specificity.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    denominator = conf_matrix[0, 0] + conf_matrix[0, 1]
    specificity = conf_matrix[0, 0] / denominator if denominator != 0 else 0
    return np.sqrt(sensitivity * specificity)


class GASel(HierarchicalEstimator):
    """
    A genetic algorithm-based feature selector for hierarchical feature spaces.

    This class implements a genetic algorithm approach based on the model proposed by
    da Silva et al., designed for optimizing feature selection in hierarchical feature
    spaces typical in high-dimensional datasets.

    Attributes
    ----------
    n_population : int
        Number of individuals in the population.
    n_generations : int
        Number of generations to evolve the population.
    mutation_prob : float
        Probability of mutating an individual's feature presence.
    she_mutation_prob : float
        Probability of a simple hierarchical elimination mutation.
    epsilon : float
        Threshold defining elitism, retaining the top epsilon fraction of individuals.
    mode : str
        Mutation mode defining the logic of mutation.
    tournament_size : int
        Size of tournament selection algorithm.
    selected_features_ : np.ndarray
        Array of indices representing the selected features after fitting the model.
    """

    def __init__(
        self,
        hierarchy=None,
        n_population=50,
        n_generations=20,
        tournament_size=5,
        mutation_prob=0.02,
        she_mutation_prob=0.3,
        epsilon=0.05,
        mode="",
    ):
        """
        Initialize the genetic algorithm-based feature selector.

        Parameters
        ----------
        hierarchy : np.ndarray, optional
            The hierarchy of the model represented as an adjacency matrix.
        n_population : int, optional
            The size of the population.
        n_generations : int, optional
            The number of generations for the genetic algorithm.
        tournament_size : int, optional
            The size of the tournament for parent selection.
        mutation_prob : float, optional
            The probability of mutating an individual gene.
        she_mutation_prob : float, optional
            The probability of applying a simple hierarchical elimination mutation.
        epsilon : float, optional
            The elitism threshold used in the genetic algorithm.
        mode : str, optional
            Whether the genetic algorithm includes Simple Hierarchical elimination.
        """
        super().__init__(hierarchy=hierarchy)
        # Hierarchy setting in both ways: nx.Digraph and adj matrix
        super()._set_hierarchy()
        self.estimator = BernoulliNB()  # Assume a default estimator if None provided
        self.n_population = n_population
        self.n_generations = n_generations
        self.tournament_size = tournament_size
        self.mutation_prob = mutation_prob
        self.she_mutation_prob = she_mutation_prob
        self.epsilon = epsilon
        self.selected_features_ = None
        self.mode = mode

    def _initialize_population(self, n_features):
        """Initialize the population with binary feature presence arrays.

        Parameters
        ----------
        n_features : int
            Number of features based on which the initial population will be generated.

        Returns
        -------
        np.ndarray
            An array representing the initial population with binary values.
        """
        return np.random.randint(2, size=(self.n_population, n_features))

    def _fitness(self, individual, X, y):
        """
        Calculate the fitness of an individual based on the geometric mean of sensitivity
        and specificity from a cross-validated model.

        Parameters
        ----------
        individual : np.ndarray
            The individual's genome representing feature inclusion as a binary array.
        X : np.ndarray
            Feature dataset.
        y : np.ndarray
            Label array.

        Returns
        -------
        float
            The calculated fitness score of the individual.
        """
        if individual.sum() == 0:
            return 0  # Return a fitness of 0 if no features are selected

        X_selected = X[:, individual == 1]
        gm_scorer = make_scorer(geometric_mean_sensitivity_specificity)
        scores = cross_val_score(
            self.estimator, X_selected, y, cv=StratifiedKFold(5), scoring=gm_scorer
        )
        penalty_for_features = 0.01 * X_selected.shape[1] / X.shape[1]
        return scores.mean() - penalty_for_features

    def _selection(self, fitness_scores):
        """
        Select parents for the next generation based on their fitness scores.

        Parameters
        ----------
        fitness_scores : np.ndarray
            Array of fitness scores from which parents will be selected.

        Returns
        -------
        np.ndarray
            Indices of the selected parents.
        """
        selected_parents = []

        for _ in range(2):
        # Randomly select k individuals for the tournament
            tournament_indices = np.random.choice(len(fitness_scores), self.tournament_size, replace=False)

            # Get the fitness scores of the tournament participants
            tournament_fitness_scores = fitness_scores[tournament_indices]

            # Find the index of the individual with the highest fitness score in the tournament
            winner_index = tournament_indices[np.argmax(tournament_fitness_scores)]

            # Add the winner to the selected parents
            selected_parents.append(winner_index)

        return np.array(selected_parents)


    def _mutation(self, individual):
        """
        Mutate an individual's genome based on the mutation probability.

        Parameters
        ----------
        individual : np.ndarray
            The individual's genome to be mutated.

        Returns
        -------
        np.ndarray
            The mutated genome.
        """
        # Simple Hierarchical Elimination Mutation Mode
        if self.mode == "she":
            for i in range(len(individual)):
                if individual[i] == 1 & (
                    self.has_ancestors(i) or self.has_descendants(i)
                ):
                    if np.random.rand() < self.she_mutation_prob:
                        individual[i] = 0
                else:
                    if np.random.rand() < self.mutation_prob:
                        individual[i] = 1 - individual[i]

        # Correlation-Based Hierarchical Mutation Mode
        elif self.mode == "cbhe":
            for i in range(len(individual)):
                if individual[i] == 1 and self.is_redundant(i):
                    corr = 0
                    ancestors = self.ancestors(i)
                    descendants = self.descendants(i)

                    for feature in ancestors:
                        corr += self.corrmatrix[i][feature]
                    for feature in descendants:
                        corr += self.corrmatrix[i][feature]
                    corr /= (len(ancestors) + len(descendants))

                    if np.random.rand() < corr:
                        individual[i] = 1 - individual[i]
                else:
                    if np.random.rand() < self.mutation_prob:
                        individual[i] = 1 - individual[i]
        # Standard Bitwise Mutation
        elif self.mode == "" or self.mode is None:
            for i in range(len(individual)):
                if np.random.rand() < self.mutation_prob:
                    individual[i] = 1 - individual[i]
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Expected 'she', 'cbhe', or ''.")

        return individual

    def _fit(self, X, y):
        """
        Run the genetic algorithm to select the best feature subset.

        Parameters
        ----------
        X : np.ndarray
            The feature dataset.
        y : np.ndarray
            The label dataset.
        """
        if self.mode == "cbhe":
            self.corrmatrix = np.corrcoef(X)

        n_features = X.shape[1]
        population = self._initialize_population(n_features)

        for _ in range(self.n_generations):
            # Fitness Scores Evaluation
            fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])

            # Elitism: Find EPS best and Add Them to the New Generation
            best_indices = fitness_scores >= np.quantile(
                fitness_scores, 1 - self.epsilon
            )
            new_population = population[best_indices].tolist()

            while len(new_population) < self.n_population:
                parent_indices = self._selection(fitness_scores)
                parent1, parent2 = (
                    population[parent_indices[0]],
                    population[parent_indices[1]],
                )
                children = _crossover(parent1, parent2)
                new_population.extend([self._mutation(child) for child in children])
            population = np.array(new_population[: self.n_population])

        fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])
        self.selected_features_ = population[np.argmax(fitness_scores)]

    def fit(self, X, y=None, columns=None):
        """
        Fit the genetic algorithm model to the data.

        Parameters
        ----------
        X : np.ndarray
            The feature dataset.
        y : np.ndarray, optional
            The label dataset.
        columns : list, optional
            Column indices that align features with the hierarchy.

        Returns
        -------
        self
            The fitted model.
        """
        self._columns = columns if columns is not None else list(range(X.shape[1]))
        self._fit(X, y)
        self._is_fitted = True
        return self

    def transform(self, X):
        """
        Transform the dataset to include only the selected features.

        Parameters
        ----------
        X : np.ndarray
            The dataset to transform.

        Returns
        -------
        np.ndarray
            The transformed dataset with only the selected features included.
        """
        if not hasattr(self, "selected_features_") or self.selected_features_ is None:
            raise ValueError("The model has not been fitted yet.")
        return X[:, self.selected_features_ == 1]

    def is_redundant(self, feature_index):
        """
        Check if a feature is redundant.

        Feature is considered to be redundant iff it has successors and/or ancestors.

        Parameters
        __________
        feature_index : int
            The feature index.

        Returns
        ___________
        bool
            Whether the feature is redundant.
        """

        if self.has_ancestors(feature_index) or self.has_descendants(feature_index):
            return True
        return False


    def ancestors(self, feature_index):
        graph = self._hierarchy
        node = feature_index
        anc = nx.ancestors(graph, node)
        anc.discard(
            "ROOT"
        )
        return anc

    def has_ancestors(self, feature_index):
        """
        Check if a feature index has any ancestors in the hierarchy.

        Parameters
        ----------
        feature_index : int
            Index of the feature to check in the hierarchy graph.

        Returns
        -------
        bool
            True if the feature has ancestors, False otherwise.
        """
        graph = self._hierarchy
        node = feature_index
        anc = nx.ancestors(graph, node)
        anc.discard(
            "ROOT"
        )  # Assuming 'ROOT' should not be considered as a valid ancestor.
        return bool(anc)

    def descendants(self, feature_index):
        graph = self._hierarchy
        node = feature_index
        return nx.descendants(graph, node)

    def has_descendants(self, feature_index):
        """
        Check if a feature index has any descendants in the hierarchy.

        Parameters
        ----------
        feature_index : int
            Index of the feature to check in the hierarchy graph.

        Returns
        -------
        bool
            True if the feature has descendants, False otherwise.
        """
        graph = self._hierarchy
        node = feature_index
        return bool(nx.descendants(graph, node))