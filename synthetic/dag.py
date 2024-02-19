# %%
import random
from enum import Enum, auto

import networkx as nx
import numpy as np


class BranchDistribution(Enum):
    DEEP = auto()
    SHALLOW = auto()
    BALANCED = auto()


import random
from typing import Dict

import networkx as nx

##TODO make branches be able to be shorter than the max depth, think about skip connections
## Joint probability von allen Paaren,
## Echten Datensatz analysieren und gucken welche zusammenhänge es tatsächlich gibt, und komplex wir die modellieren müssen


def generate_dag(
    num_features: int = 10,
    max_depth: int = 6,
    branch_distribution: BranchDistribution = BranchDistribution.BALANCED,
    tree_likeness: float = 0.5,
    branch_termination_prob: float = 0.1,
    skip_connection_prob: float = 0.1,
) -> nx.DiGraph:
    """
    Generate a random DAG with options for branches shorter than max depth and skip connections.

    Parameters:
    - num_features: Total number of features (nodes) to generate.
    - max_depth: Maximum depth of the DAG.
    - branch_distribution: Distribution of branches (DEEP, SHALLOW, BALANCED).
    - tree_likeness: Probability to keep tree structure, less than this value creates extra parents (skip connections).
    - branch_termination_prob: Probability to terminate a branch before the max depth.
    - skip_connection_prob: Probability to add skip connections between non-consecutive nodes.
    """
    if num_features <= 0:
        raise ValueError("num_features must be greater than 0")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if not 0 <= tree_likeness <= 1:
        raise ValueError("tree_likeness must be between 0 and 1")
    if not 0 <= branch_termination_prob <= 1:
        raise ValueError("branch_termination_prob must be between 0 and 1")
    if not 0 <= skip_connection_prob <= 1:
        raise ValueError("skip_connection_prob must be between 0 and 1")

    dag = nx.DiGraph()
    dag.add_node(0)  # start with a root node
    current_level = 0
    level_nodes: Dict[int, list[int]] = {0: [0]}

    while len(dag.nodes) < num_features:
        new_level_nodes = []
        if current_level >= max_depth:
            break

        branching_factor = determine_branching_factor(
            branch_distribution, current_level, max_depth
        )

        for node in level_nodes[current_level]:
            # Randomly decide whether to terminate this branch earlier
            if random.random() < branch_termination_prob and current_level > 1:
                continue

            for _ in range(branching_factor):
                if len(dag.nodes) < num_features:
                    new_node = len(dag.nodes)
                    dag.add_node(new_node)
                    dag.add_edge(node, new_node)
                    new_level_nodes.append(new_node)

                    if tree_likeness < random.random():
                        possible_parents = [
                            node
                            for level in level_nodes.values()
                            for node in level
                            if level != current_level
                        ]
                        if possible_parents:
                            extra_parent = random.choice(possible_parents)
                            if extra_parent != new_node and not nx.has_path(
                                dag, new_node, extra_parent
                            ):
                                dag.add_edge(extra_parent, new_node)

        level_nodes[current_level + 1] = new_level_nodes
        current_level += 1

        # Add skip connections
        if random.random() < skip_connection_prob:
            possible_parents = [
                node
                for level in level_nodes.values()
                for node in level
                if level != current_level
            ]
            if possible_parents:
                child = random.choice(new_level_nodes)
                parent = random.choice(possible_parents)
                if not nx.has_path(dag, child, parent):  # Check to avoid cycles
                    dag.add_edge(parent, child)

    last_level = max(level_nodes.keys())
    while len(dag.nodes) < num_features:
        new_nodes = []
        for node in level_nodes[last_level]:
            if len(dag.nodes) >= num_features:
                break  # Exit if we have reached the desired number of nodes
            new_node = len(dag.nodes)
            dag.add_node(new_node)
            dag.add_edge(node, new_node)
            new_nodes.append(new_node)

        if new_nodes:
            # Update level_nodes with new nodes if they were added
            last_level += 1
            level_nodes[last_level] = new_nodes
        else:
            # If no new nodes were added (e.g., all branches were terminated), add a new node to an existing node
            # This is a fallback to ensure num_features is met, which should rarely be necessary
            existing_node = random.choice(list(dag.nodes))
            new_node = len(dag.nodes)
            dag.add_node(new_node)
            dag.add_edge(existing_node, new_node)

    return dag


def determine_branching_factor(
    branch_distribution: BranchDistribution, current_level: int, max_depth: int
) -> int:
    # This function decides the branching factor based on the distribution type
    if branch_distribution == BranchDistribution.DEEP:
        return min(2 + current_level, max_depth)
    elif branch_distribution == BranchDistribution.SHALLOW:
        return max(max_depth - current_level, 1)
    elif branch_distribution == BranchDistribution.BALANCED:
        return 2
    else:
        raise ValueError(f"Unknown branch distribution: {branch_distribution}")


def generate_synthetic_dataset(
    num_samples: int,
    num_features: int,
    max_depth: int,
    branch_distribution: BranchDistribution,
    tree_likeness: float,
) -> tuple[np.ndarray, np.ndarray, nx.DiGraph]:
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
    """
    Calculate the influence of a feature based on its position and connectivity in the DAG.
    More central nodes or those with more connections might have higher influence.
    """
    descendants = len(list(nx.descendants(dag, node)))
    ancestors = len(list(nx.ancestors(dag, node)))
    centrality = nx.betweenness_centrality(dag).get(
        node, 0
    )  # Node centrality as a proxy for importance

    # Influence based on descendants, ancestors, and centrality
    influence = 0.5 + 0.05 * descendants + 0.05 * ancestors + centrality
    return min(influence, 1)


def generate_labels(X: np.ndarray, dag: nx.DiGraph) -> np.ndarray:
    """
    Generate labels using a combination of features influenced by the DAG structure.
    The influence is determined by the node's centrality and the number of descendants,
    indicating the importance of a feature in the DAG.
    """
    # Calculate feature influences based on the DAG structure
    influences = np.array(
        [calculate_feature_influence(node, dag) for node in range(X.shape[1])]
    )

    # Adjust features by their influences
    adjusted_features = X * influences

    # Simple example of using adjusted features to generate labels
    # Here, we sum the influenced features and add a non-linear transformation
    scores = np.dot(adjusted_features, np.ones(X.shape[1]))
    non_linear_scores = 1 / (
        1 + np.exp(-scores)
    )  # Sigmoid function for a non-linear effect

    # Binarize labels based on a threshold
    labels = (non_linear_scores > 0.5).astype(int)

    return labels
