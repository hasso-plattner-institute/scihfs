def count_subsets(graph, v, included, dp):
    if dp[v][included] != -1:
        return dp[v][included]

    # Count subsets without including this vertex
    exclude_count = count_subsets(graph, v + 1, included, dp) if v + 1 < len(graph) else 1

    # Count subsets including this vertex
    include_count = 0
    if not included:
        include_count = count_subsets(graph, v + 1, True, dp) if v + 1 < len(graph) else 1
        for u in graph[v]:
            include_count *= count_subsets(graph, u, False, dp)

    dp[v][included] = exclude_count + include_count
    return dp[v][included]

def subsets_in_dag(graph):
    n = len(graph)
    dp = [[-1] * 2 for _ in range(n + 1)]

    return count_subsets(graph, 0, False, dp)

# Example usage:
graph = {
    0: [1, 2],
    1: [3],
    2: [3],
    3: []
}
print(subsets_in_dag(graph))  # Output: 6
