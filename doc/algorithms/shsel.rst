############
SHSEL & HFE
############

SHSEL
======

The SHSEL feature selection method was introduced by Ristoski and Paulheim in 2014 :cite:`ristoski2014feature`.
There are two parts to the algorithm.
The first one is the initial selection. For this part, first, all leaf nodes of the hierarchy tree are identified. Then,
starting from each leaf node the tree is traversed upwards and the relevance similarity value between the current node and
the parent node is calculated. If the similarity is larger than a certain threshold the child node is removed from the feature space.
When information gain is used as the relevance metric, the relevance values are normalized to the range [0, 1] (following the paper's
use of RapidMiner's normalized information-gain weights).

The second part of the algorithm is the pruning part. Here, all paths from leaf to root node are identified in the hierarchy tree.
Then we calculate the average information gain for each path and remove all features with an information gain lower than this average
value from the feature space. If a feature is on more than one path its information gain only needs to be higher than the average
information gain on at least one of the paths.


This algorithm has four hyperparameters. The ``relevance_metric`` used in the initial selection can be either correlation or
information gain. The ``similarity_threshold`` is the cut-off used in the initial selection; if left as ``None`` (the default)
it resolves to the paper's metric-specific defaults, i.e. 0.99 for information gain and 0.6 for correlation. ``ig_average``
controls how the per-path information-gain average is computed in the pruning stage: ``"full_path"`` (the default)
averages over every node on the path, including nodes already removed in the initial selection,
while ``"survivors_only"`` averages over the surviving features only; in both cases though, only features
that have already survived the initial selection can be selected here.
Finally, ``pruning`` (default ``True``) toggles the pruning stage (Algorithm 2): with ``pruning=False`` only the initial
selection (Algorithm 1)is applied, corresponding to the paper's *initialSHSEL*, while
``pruning=True`` runs both stages (Algorithms 1 and 2, equivalent to *pruneSHSEL*).

HFE -- currently NOT supported (!)
====
The hierarchical feature engineering method (HFE) was proposed by Oudah and Henschel in 2018 :cite:`oudah2018taxonomy` and is an extension
of Ristoski and Paulheim's SHSEL algorithm.  SHSEL is meant for binary features while the HFE method expects numerical
features. Before starting the feature selection algorithm the features are preprocessed by adding the descendants' values
of each node to the own value. These sums are used for the following feature selection algorithm. For the initial selection
part of the SHSEL algorithm correlation is used as the relevance metric. After the pruning part, a third part is added to the
algorithm. From the remaining nodes, we select all leaves of incomplete paths. Incomplete paths are paths that are shorter
than the maximum path. From these leaves, all are removed from the selected features that have a lower information gain score
than the global average or an information gain of 0.

To use the HFE extension, the relevance ``similarity`` parameter of the SHELSelector needs to be set to correlation and the
``use_hfe_extension`` parameter needs to be set to True.
