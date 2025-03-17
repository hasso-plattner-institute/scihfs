###########
Experiments
###########

In order to evaluate the implemented methods and compare them with each other we conducted a number of experiments.
To allow further comparisons to the papers introducing the algorithms we focused on following their set-up.

Datasets
=========

For the purpose of evaluating the implemented methods we use two different datasets, following the mentioned papers about lazy and eager learning approaches. 
For lazy learning we use a bioinformatic dataset and for the eager approach the SportsTweets dataset. 

Bioinformatic dataset
**********************

For the purpose of evaluating the lazy methods we follow Wan, Freitas and de Magalhaes' task of predicting a *C. elegans* gene’s effect on the organism’s longevity. 
The corresponding information is obtained from the database of Human Ageing Genomic Resources :cite:p:`TacutuRobi2013HAGR`.
Hierarchical relations are pictured in the Gene Ontology (GO) :cite:p:`GeneOntologyConsortium2004` from the National Institutes of Health. 

The GO collects terms and hierarchical relationships in a "is-a" generalizations/specializations, among those, building a directed acyclic graph.
If a gene is associated with the GO term i, it is also associated with all ancestors of i in the hierarchy (assured in the preprocessing of instances).
The gene2go :cite:p:`gene2go` maps the GO-terms to the associated genes.
We build our dataset by selecting all EntrezID numbers of each *C. elegans* gene in the HAGR database with the same EntrezID number in the NCBI gene database gene2go [2023-06-25]. 
Then the GO terms for matched genes are mapped to the data from the gene2go database. Thus, we obtain a dataset with 889 genes (rows) and 27416 go-terms (columns) set to 1 if the go-term is present in this gene.
After deleting genes associated with no go-terms the dataset includes 819 genes and the final set with only genes which effect on longevity is known, the dataset contains 793 rows.

SportsTweets
************

The SportsTweets dataset [1]_ which we primarily use for evaluating the eager prediction methods is inspired by the *Sports Tweets T* dataset used
in the paper proposing the SHSEL Feature Selection algorithm by :cite:authors:`ristoski2014feature`. It consists of tweets which are classified into
sports related and non-sports related tweets. The dataset is linked to DBpedia using the 
DBpedia Spotlight service. For the feature selection and classification tasks we use the extracted DBpedia entitites and types which we generate
for the entitites using the `direct_type_generator <https://kgextension.readthedocs.io/en/latest/source/usage_generators.html#direct-type-generator>`_ 
from the `kgextension library <https://github.com/om-hb/kgextension>`_. This method also generates the corresponding hierachy which is 
build using hyponymy and hypernymy relationships between the types. For computational reasons we only use a subset including the first 300 samples 
of the dataset. 


Experiment Setup and Evaluation
=================================

Eager Learning
***************
For the eager learning approach we evaluate on both datasets. We split the dataset into 70% for training and 30% for evaluation. 
We then fit the feature selectors on the training set and transform both training and eval set using the features selected on 
the training set. For all feature selectors the default hyperparameters are used. These default values are either based on the 
papers in which the methods are proposed or chosen heuristically if the papers don't mention which values should be used.
After feature selection, we fit a `Naive Bayes classifier <https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html>`_ 
implemented in scikit-learn. Here we also keep the default hyperparameters. We finally evaluate the classifier on the eval set.
We evaluated the effects of the feature selection by computing the classification accuracy and the feature compression rate. The 
scores are compared with a baseline where we classify the dataset without performing feature selection.


Lazy Learning
**************

For the purpose of evaluating the methods on the real-world datasets and comparing the results with the papers 
proposing the methods, we adopt their metrics to build our prediction scores.
First, we use accuracy to evaluate the classification performance.
The predictions in the lazy learning methods are obtained by a Naive Bayes, which is already implemented in the methods.
Secondly, we measure the performance of feature selection by the compression of the features, so the ratio 
of selected features to all features.

The hyperparameter k for lazy learning, that is the maximum number of selected features, was chosen in 
accordance with the papers with k = [30, 40, 50].
We evaluate the lazy learning methods on the gene dataset. It was divided into a train-test set on a ratio of 70/30, since the dataset is relatively 
small and the lazy learning approach is executed per testing instance, which should not be too small.

Results
========
.. csv-table:: Eager Learning
   :file: eagertable.csv
   :widths: 10,10,10,10,10,10,10
   :header-rows: 1

The values in brackets are the results from the :cite:authors:`ristoski2014feature` paper.

.. csv-table:: Lazy Learning
   :file: lazytable.csv
   :header-rows: 1

.. csv-table:: 
   :file: lazytable2.csv
   :header-rows: 1

The values in brackets are the results from :cite:authors:`hnb` and others.

Discussion
==========

Eager Learning
**************
For the eager approach we used a dataset that was also used in the paper by Ristoski and Paulheim :cite:p:`ristoski2014feature`. 
However, we only used a subset of the dataset and achieved very different results even when only classifying the 
complete feature set without performing feature selection. Therefore, it is difficult comparing the results. In the 
table, we can see the results from the paper in brackets. The accuracy values from the paper are higher for all approaches. 
Moreover, in the paper, the SHSEL algorithm achieves the best accuracy score whereas in our experiments the GTD approach had 
the best result. We suspect that despite our dataset being based on the same data it is too different from the dataset used 
in the paper. We only use a subset and we had to create the hierarchy ourselves and cannot confirm if it was created in the 
exact same way as the hierarchy used in the experiments for the paper. 

Instead of comparing the results with the paper it is more interesting to compare the different approaches to each other 
and the baseline. Our results show that all feature selection approaches achieved a slightly better classification result 
than the baseline. Additionally, we can see that lower compression rates, meaning fewer features, result in higher accuracy 
scores. This shows us that eliminating redundant features can affect classification performance.

When performing experiments we also noticed drastic differences in computation time between the 
different algorithms. For the Hill Climbing Top Down approach we could not obtain any results because 
the feature selection did not finish in a reasonable time. As we did not focus on exploring especially 
efficient ways of implementing the algorithms we expect there to be some optimizations possible that would 
decrease the computation time.

Lazy Learning
**************

Regarding the experiments with the lazy learning approach, we see, that the feature selection has less impact on the prediction results.
While the accuracy without any feature selection is 67.20%, the accuracy of the other methods is equal or worse.
Those results differ from the results in the paper, which are given in brackets. They claim to achieve better results using the feature selection.

The further consideration of the rather similar accuracy values we have obtained suggests, that the Naive Bayes constantly predicts the same value.
We verify this assumption with the calculation of the recall and precision.
Since the recall of the postive class is nearly 0, the Naive Bayes is not learning.
Computing the proportion of positive and negative occurences, we get a value near the precision scores, so the estimator chosen in the papers does not 
fit to the used dataset.
Especially for the methods HNB and RNB which allow to restrict the number of chosen feature resulting in very small compression rates, 
the Naive Bayes predicts the negative class almost always.

Hence, we repeated the experiments with a Gaussian Naive Bayes and a Decision Tree, but obtained similar predictions seeming 
like the classifier has not learned from the features.

We assume that the presented feature selection approaches - filtering out a lot of data - may be more valuable in larger datasets.


Conclusion
==========

For both eager and lazy learning approaches we could not reproduce the results from the papers.. However, this does 
not necessarily mean that the feature selection approaches do not work as well as the papers suggest. Small differences in implementation, configuration or 
in the dataset can be the reason for our results. We do not have the exact same datasets or access to the original code so we do not 
know all aspects in which our experiments differed. 
Still, we achieved positive results with the eager learning approaches and found challenges like algorithms with high computational complexity
which can be addressed in the future.

.. [1] Downloaded from https://data.dws.informatik.uni-mannheim.de/rmlod/LOD_ML_Datasets/data/datasets/SportTweets/ (2nd July 2023)
