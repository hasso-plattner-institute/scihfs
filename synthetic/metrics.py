# %%
import networkx as nx


def average_in_degree(G):
    """Calculates the average in-degree of the graph."""
    in_degrees = dict(G.in_degree())  # Dictionary of in-degrees
    avg_in_degree = sum(in_degrees.values()) / len(in_degrees)
    return avg_in_degree


def average_out_degree(G):
    """Calculates the average out-degree of the graph."""
    out_degrees = dict(G.out_degree())  # Dictionary of out-degrees
    avg_out_degree = sum(out_degrees.values()) / len(out_degrees)
    return avg_out_degree


def average_degree(G):
    """Calculates the overall average degree (in + out) of the graph."""
    # Since each edge contributes to one in-degree and one out-degree, we can simply
    # use the total number of edges * 2 divided by the number of nodes to get the overall average degree
    total_degrees = 2 * len(
        G.edges()
    )  # Each edge contributes to both an in-degree and an out-degree
    avg_degree = total_degrees / len(G.nodes())
    return avg_degree


def longest_dag_path(G):
    """Finds the longest path in a DAG."""
    # Ensure the graph is a DAG
    if nx.is_directed_acyclic_graph(G):
        return nx.dag_longest_path(G)
    else:
        raise ValueError("Graph is not a Directed Acyclic Graph (DAG)")


def describe_dag(dag: nx.DiGraph) -> dict:
    return {
        "num_nodes": len(dag.nodes),
        "num_edges": len(dag.edges),
        "average_in_degree": average_in_degree(dag),
        "average_out_degree": average_out_degree(dag),
        "average_degree": average_degree(dag),
        "density": nx.density(dag),
        "longest_path": longest_dag_path(dag),
        "average_path_length": nx.average_shortest_path_length(dag),
        "betweenness_centrality": nx.betweenness_centrality(dag),
        "closeness_centrality": nx.closeness_centrality(dag, reversed=True),
        "clustering_coefficient": nx.average_clustering(dag),
    }


# %%
G = nx.read_gml("data/go_dag.gml")

describe_dag(G)
# %%
