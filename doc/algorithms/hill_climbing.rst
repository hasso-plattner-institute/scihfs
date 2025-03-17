##############
Hill Climbing
##############

Wang et al. proposed two hill-climbing feature selection methods. The top-down approach was introduced in 
2002 :cite:`wang2002learning` and the bottom-up approach is an updated version proposed in 2003 :cite:`wang2003comparative`. 

Top-Down Hill Climbing 
========================
For the top-down approach, we start with an optimal concept set that includes only the root node. 
Then we iterate over the optimal concept set. We keep replacing nodes with their children and calculate the 
fitness of the resulting concept sets to finally find the optimal concept set.

The fitness function is calculating distances between all samples. It maximizes the distances between 
samples of different classes and minimizes distances between samples of the same class. For this fitness function, 
the hyperparameter ``alpha`` is needed.

Bottom-Up Hill Climbing
========================
For the bottom-up approach, we start with all leaves of the hierarchy tree as the current concept set. 
Then we keep taking unvisited nodes of this set and replacing them with their parent node. All children of this 
parent node are removed from the set. If this concept set has a higher fitness than the original one we go on with this set. 

This feature selection was a different fitness and a different distance function than the top-down hill climbing approach. 
Instead of the distance function cosine similarity is used. The fitness function needs an additional hyperparameter ``k`` because 
it looks at the k nearest neighbors for each sample.