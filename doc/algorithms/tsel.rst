#######
TSEL
#######

The tree-based feature selection algorithm (TSEL) was introduced by Jeong and Myaeng in 2013 :cite:`jeong2013feature`. 
To select the most relevant features first the most representative feature for each path in the hierarchy graph is 
selected. The relevance is determined using the lift metric here. Then The selected features are filtered further by 
checking if any representative has descendants that were also selected. If a node has descendants that were also selected 
it is removed from the representatives. Finally, 
:math:`\chi^2`-feature selection is performed on the representative to determine 
the final selected features.

We implemented the selection of the representative feature for each path in two different ways because it was not entirely 
clear from the paper how this part should be implemented. The user can select which algorithm should be used.