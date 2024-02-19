# %%
import itertools
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from synthetic.dag import BranchDistribution, generate_dag, generate_synthetic_dataset


def visualize_dag(dag: nx.DiGraph, ax=None):
    """
    Visualize the DAG with nodes placed according to their hierarchical level, centered in the middle.
    """
    # Calculate levels based on the longest path from roots
    if ax is None:
        ax = plt.gca()  # Get the current active axes

    levels = calculate_node_levels(dag)

    # Group nodes by level
    level_groups = defaultdict(list)
    for node, level in levels.items():
        level_groups[level].append(node)

    # Determine the width of the plot based on the widest level
    max_width = max(len(nodes) for nodes in level_groups.values())

    # Prepare position dict for nx.draw(), centering nodes within each level
    final_pos = {}
    for level, nodes in level_groups.items():
        level_width = len(nodes)
        offset = (max_width - level_width) / 2  # Calculate offset to center nodes
        for i, node in enumerate(sorted(nodes)):  # Sort for consistent ordering
            final_pos[node] = (
                i + offset,
                -level,
            )  # Adjust x position by offset, negative level for top-down

    # Drawing
    nx.draw(
        dag,
        final_pos,
        ax=ax,
        with_labels=True,
        arrows=True,
        node_size=600,
        node_color="lightblue",
        font_size=10,
    )


def calculate_node_levels(dag):
    """
    Calculate node levels based on the longest path from any root.
    """
    levels = {}
    for node in nx.topological_sort(dag):
        preds = list(dag.predecessors(node))
        if preds:
            levels[node] = max([levels[pred] for pred in preds]) + 1
        else:
            levels[node] = 0
    return levels


# Visualization functions for the dataset
def visualize_feature_distribution(X: np.ndarray):
    """
    Visualize the distribution features
    """
    plt.figure(figsize=(12, 8))
    for i in range(X.shape[1]):
        plt.subplot(3, 4, i + 1)
        plt.hist(X[:, i], bins=30, color="lightblue", edgecolor="black")
        plt.title(f"Feature {i}")
    plt.tight_layout()
    plt.show()


def visualize_feature_interaction(X: np.ndarray, y: np.ndarray):
    """
    Visualize the interaction between two binary features.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="skyblue", label="Class 0", alpha=0.7)
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="salmon", label="Class 1", alpha=0.7)
    plt.title("Feature Interaction")
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1")
    plt.legend()
    plt.show()


# Generate the synthetic dataset and DAG
X, y, dag = generate_synthetic_dataset(
    num_samples=100,
    num_features=10,
    max_depth=5,
    branch_distribution=BranchDistribution.BALANCED,
    tree_likeness=0.5,
)

# Visualize the DAG
visualize_dag(dag)

# Visualize the distribution of the first feature
visualize_feature_distribution(X)

# Visualize the interaction between the first two features
visualize_feature_interaction(X, y)


# %%
# TODO Multiple roots?
# Parameter ranges (example values, adjust as needed)
def small_multiples(
    num_features_options=[10, 20],
    max_depth_options=[2, 3, 6, 7],
    branch_distribution_options=[
        BranchDistribution.DEEP,
        BranchDistribution.SHALLOW,
        BranchDistribution.BALANCED,
    ],
    tree_likeness_options=[0.2, 0.5, 0.8],
):

    # Create a figure for the small multiples
    fig, axes = plt.subplots(
        len(num_features_options) * len(max_depth_options),
        len(branch_distribution_options) * len(tree_likeness_options),
        figsize=(60, 30),
        dpi=300,
    )

    # Flatten axes array for easy indexing
    axes = axes.flatten()

    # Iterate over all combinations of parameters
    for idx, (num_features, max_depth, branch_distribution, tree_likeness) in enumerate(
        itertools.product(
            num_features_options,
            max_depth_options,
            branch_distribution_options,
            tree_likeness_options,
        )
    ):
        # Generate DAG
        dag = generate_dag(num_features, max_depth, branch_distribution, tree_likeness)

        ax = axes[idx]
        plt.sca(ax)

        visualize_dag(dag, ax=ax)

        ax.set_title(
            f"Features: {num_features}, Depth: {max_depth}\nBranch: {branch_distribution.name}, Tree-likeness: {tree_likeness}"
        )

    # Adjust layout
    plt.tight_layout()
    plt.show()


small_multiples()

# %%
