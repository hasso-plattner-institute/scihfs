import numpy as np
import networkx as nx
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_X_y
from sklearn.model_selection import cross_val_score

from hfs.selectors import HierarchicalEstimator


class gasel(HierarchicalEstimator):
    def __init__(self, hierarchy=None, estimator=None, n_population=50, n_generations=100, crossover_prob=0.8,
                 mutation_prob=0.02):
        super().__init__(hierarchy=hierarchy)
        self.estimator = estimator
        self.n_population = n_population
        self.n_generations = n_generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.selected_features_ = None

    def _initialize_population(self, n_features):
        return np.random.randint(2, size=(self.n_population, n_features))

    def _fitness(self, individual, X, y):
        if individual.sum() == 0:
            return 0
        X_selected = X[:, individual == 1]
        scores = cross_val_score(self.estimator, X_selected, y, cv=5)
        return scores.mean()

    def _selection(self, fitness_scores):
        parents_indices = np.argpartition(fitness_scores, -2)[-2:]
        return parents_indices

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

    def _cbhe_mutation(self, individual):
        """Correlation-Based Mutation procedure"""
        pass

    def _she_mutation(self, individual):
        """Simple Hierarchical Elimination Mutation procedure"""
        pass

    def fit(self, X, y=None, columns=None):
        super().fit(X, y, columns)
        X, y = check_X_y(X, y)
        n_features = self.n_features_in_
        population = self._initialize_population(n_features)

        for generation in range(self.n_generations):
            fitness_scores = np.array([self._fitness(ind, X, y) for ind in population])
            new_population = []
            while len(new_population) < self.n_population:
                parent_indices = self._selection(fitness_scores)
                parent1, parent2 = population[parent_indices]
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutation(child1)
                child2 = self._mutation(child2)
                new_population.extend([child1, child2])
                population = np.array(new_population)[:self.n_population]
                best_score_idx = np.argmax(fitness_scores)
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
