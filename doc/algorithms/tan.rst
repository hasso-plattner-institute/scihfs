##############################
Tree-based redundancy removal
##############################


The last method implemented is a Hierarchical Redundancy Eliminated Tree.
Augmented Naive Bayes (HRE–TAN), which is slightly modified in the library.
The algorithm builds a minimal spanning tree (MST), by adding all possible edges (that is all possible feature pairs :math:`f_i` and :math:`f_j`),
 that meets certain conditions to an undirected graph (UDAG).
First, the edges are sorted in descending order of their conditional mutual information TODO: link and then added in this order, if:
1. :math:`f_i` and :math:`f_j`are not hierarchically redundant in the current test instance
2. adding the edge between :math:`f_i` and :math:`f_j` does not introduce a cyle in the UDAG
3. the edge is still available.
An edge :math:`e=(f_h, f_g)` becomes unavailable if there is already an edge :math:`(f_i, f_j)`
such that either :math:`f_i` or :math:`f_j` is an ancestor or descedant as a member of :math:`e` that has the same value.
The UDAG is then turned in an MST by selecting any node as root node and marking the direction of all edges from it outwards to other vertices.
Training and test set are then reduced to the features obtained with the MST and fed into a lazy TAN model.
In contrast to their paper we give the obtained datasets to a customer estimator, which is per default a naive bayes.