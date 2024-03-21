import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.utils.validation import check_array, check_X_y
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, confusion_matrix
from hfs.selectors import HierarchicalEstimator


class gasel(HierarchicalEstimator):
    """Genetic Algorithm for Hierarchical Feature Selection"""
    #to think about the default size and number of generations
    #in general: problem of default values choosing
    def __init__(self, hierarchy=None, estimator=BernoulliNB(), n_population=50, n_generations=100, crossover_prob=0.8,
                 mutation_prob=0.02, she_mutation_prob=0.3, epsilon=0.15):
        super().__init__(hierarchy=hierarchy)
        self.estimator = estimator
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.she_mutation_prob = she_mutation_prob
        self.selected_features_ = None
        self.epsilon = epsilon

    def _initialize_population(self, n_features):
        return np.random.randint(2, size=(self.n_population, n_features))

    def geometric_mean_sensitivity_specificity(self, y_true, y_pred):
        """The Geometric Mean of Sensitivity and Specificity"""
        conf_matrix = confusion_matrix(y_true, y_pred)
        sensitivity = recall_score(y_true, y_pred)
        specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1])
        return np.sqrt(sensitivity * specificity)

    def _fitness(self, individual, X, y):
        #Individuals with no features selected are to be excluded
        if individual.sum() == 0:
            return 0
        X_selected = X[:, individual == 1]
        # Scorer for GM
        gm_scorer = make_scorer(self.geometric_mean_sensitivity_specificity)

        # X-Validation procedure
        scores = cross_val_score(self.estimator, X_selected, y, cv=StratifiedKFold(5), scoring=gm_scorer)

        # Penalization for redundant features -- to be changed
        penalty_for_features = 0.01 * X_selected.shape[1] / X.shape[1]  # Proportional decrease
        return scores.mean() - penalty_for_features

    def _selection(self, fitness_scores):
        """Parents Selection"""
        #parents_indices = np.argpartition(fitness_scores, -2)[-2:]
        #return parents_indices
        parents_indices = np.random.choice(len(fitness_scores), 2, replace=False)
        return parents_indices

    def _select_epsilon(self, population, X, y):
        fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])
        threshold = np.quantile(fitness_scores, 1 - self.epsilon)
        selected_indices = [i for i, score in enumerate(fitness_scores) if score >= threshold]
        return [population[i] for i in selected_indices]

    def _crossover(self, parent1, parent2):
        point = np.random.randint(len(parent1))
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def _mutation(self, individual):
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                individual[i] = 1 - individual[i]
        return individual

    #To be revised
    def _is_redundant(self, feature):
        node = self._columns[feature]

        for anc in self._hierarchy.predecessors(node):
            if self.selected_features_[anc] == 1:
                return True
        for des in self._hierarchy.successors(node):
            if self.selected_features_[des] == 1:
                return True
        return False

    def _she_mutation(self, individual):
        for i in range(len(individual)):
            if individual[i] and self._is_redundant(individual[i]):
                if np.random.rand() < self.she_mutation_prob:
                    individual[i] = 1 - individual[i]
        return individual
    def _cbhe_mutation(self, individual):
        """Correlation-Based Mutation procedure"""
        pass

    def fit(self, X, y=None, columns=None):
        #to check if it meets the sklearn requirements fro naming
        #to add another stopping criteria
        super().fit(X, y, columns)
        X, y = check_X_y(X, y)
        n_features = self.n_features_in_
        print(n_features)

        population = self._initialize_population(n_features)
        print(len(population))
        for generation in range(self.n_generations):
            fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])
            print(len(fitness_scores))
            new_population = self._select_epsilon(population, X, y)
            print(fitness_scores)
            while len(new_population) < self.n_population:
                parent_indices = self._selection(fitness_scores)
                parent1, parent2 = population[parent_indices]
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                new_population.extend([child1, child2])
            #to be checked
            population = np.array(new_population)[:self.n_population]

                #new population is not yet evaluated !!!
                #reevaluate and revice again
            best_score_idx = np.argmax(fitness_scores)
            print(best_score_idx)
            best_score = fitness_scores[best_score_idx]
            print(f"Generation {generation + 1}, Best Fitness: {best_score}")

        # Selecting the best individual
        final_fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])
        best_individual_index = np.argmax(final_fitness_scores)
        self.selected_features_ = population[best_individual_index]

        return self

    def transform(self, X):
        """
        Transform X to include only the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("Fit the model before calling transform.")

        X_transformed = super().transform(X)
        return X_transformed[:, self.selected_features_ == 1]

    def get_feature_support(self):
        """
        Returns a mask of the selected features.
        """
        if self.selected_features_ is None:
            raise ValueError("Fit the model before calling get_feature_support.")
        return self.selected_features_ == 1
