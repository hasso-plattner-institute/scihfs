##################
Greedy Top Down
##################

The Greedy Top Down (GTD) feature selection method was proposed by Lu et al. in 2013 :cite:`lu2013domain`. In this algorithm, we iterate 
over the nodes on the first level after the root node. For each of these nodes, we get all descendants and sort them 
according to their information gain ratio. Then comes the pruning step. For this step, we iteratively remove the first 
element of the sorted list and add it to the selected features. Then we remove all ancestors and descendants of this 
node and continue with the next element from the front of the list.

We added an optional variation to this algorithm. The GTD approach expects the hierarchy to have a tree structure. 
If it is not a tree but a DAG the parameter ``iterate_first_level`` can be set to False. Then instead of iterating the 
nodes on the first level we only look at the descendents of the root node. This can achieve similar behavior to the 
original algorithm for some DAG but it should still be kept in mind that the original algorithm was designed for trees.