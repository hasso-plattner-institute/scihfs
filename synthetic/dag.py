import networkx as nx
import random
import numpy as np

from enum import Enum, auto

class BranchDistribution(Enum):
    DEEP = auto()
    SHALLOW = auto()
    BALANCED = auto()


import networkx as nx
import random
from typing import Dict

def generate_dag(num_features: int, max_depth: int, branch_distribution: BranchDistribution, tree_likeness: float) -> nx.DiGraph:
    """
    Generate a random DAG based on given parameters.
    """
    dag = nx.DiGraph()
    dag.add_node(0)  # start with a root node

    # Variables to keep track of the current level and nodes at each level
    current_level = 0
    level_nodes: Dict[int, list[int]] = {0: [0]}  # level: [nodes]

    while len(dag.nodes) < num_features:
        new_level_nodes = []

        # Determine branching factor based on branch_distribution
        branching_factor = determine_branching_factor(branch_distribution, current_level, max_depth)

        for node in level_nodes[current_level]:
            for _ in range(branching_factor):
                if len(dag.nodes) < num_features:
                    new_node = len(dag.nodes)
                    dag.add_node(new_node)
                    dag.add_edge(node, new_node)
                    new_level_nodes.append(new_node)

                    if tree_likeness < random.random():
                        extra_parent = random.choice(list(dag.nodes))
                        if extra_parent != new_node:
                            dag.add_edge(extra_parent, new_node)

        level_nodes[current_level + 1] = new_level_nodes
        current_level += 1

        # Break if max depth is reached
        if current_level == max_depth:
            break

    return dag

def determine_branching_factor(branch_distribution: BranchDistribution, current_level: int, max_depth: int) -> int:
    # This function decides the branching factor based on the distribution type
    if branch_distribution == BranchDistribution.DEEP:
        return min(2 + current_level, max_depth)
    elif branch_distribution == BranchDistribution.SHALLOW:
        return max(max_depth - current_level, 1)
    elif branch_distribution == BranchDistribution.BALANCED:
        return 2
    else:
        raise ValueError(f"Unknown branch distribution: {branch_distribution}")
    

def generate_dag(num_features: int, max_depth: int, branch_distribution: BranchDistribution, tree_likeness: str) -> nx.DiGraph:
    """
    Generate a random DAG based on given parameters.
    """
    dag = nx.DiGraph()
    dag.add_node(0)  # start with a root node

    # Variables to keep track of the current level and nodes at each level
    current_level = 0
    level_nodes: Dict[int, list[int]] = {0: [0]}  # level: [nodes]

    while len(dag.nodes) < num_features:
        new_level_nodes = []

        # Determine branching factor based on branch_distribution
        branching_factor = determine_branching_factor(branch_distribution, current_level, max_depth)

        for node in level_nodes[current_level]:
            for _ in range(branching_factor):
                if len(dag.nodes) < num_features:
                    new_node = len(dag.nodes)
                    dag.add_node(new_node)
                    dag.add_edge(node, new_node)
                    new_level_nodes.append(new_node)

        level_nodes[current_level + 1] = new_level_nodes
        current_level += 1

        # Break if max depth is reached
        if current_level == max_depth:
            break

    return dag



def generate_synthetic_dataset(num_samples: int, num_features: int, max_depth: int, branch_distribution: BranchDistribution, tree_likeness: float) -> tuple[np.ndarray, np.ndarray, nx.DiGraph]:
    """
    Generate a synthetic dataset based on a DAG.
    """
    dag = generate_dag(num_features, max_depth, branch_distribution, tree_likeness)

    # Initialize the feature matrix and label vector
    X = np.zeros((num_samples, num_features))
    y = np.zeros(num_samples)

    # Populate the feature matrix based on the DAG
    for node in dag.nodes:
        feature_influence = calculate_feature_influence(node, dag)
        X[:, node] = np.random.binomial(1, feature_influence, num_samples)

    # Generate labels
    y = generate_labels(X, dag)

    return X, y, dag

def calculate_feature_influence(node: int, dag: nx.DiGraph) -> float:
    descendants = len(list(nx.descendants(dag, node)))
    return min(0.5 + descendants * 0.05, 1)

def generate_labels(X: np.ndarray, dag: nx.DiGraph) -> np.ndarray:
    return np.dot(X, np.random.rand(X.shape[1])) > X.shape[1] * 0.5