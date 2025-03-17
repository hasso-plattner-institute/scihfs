####################
HNB, HNB-s, RNB
####################

2013 Wan, Freitas and de Magalhaes introduced an algorithm to select features based on their relevance and hierarchical redundancy (:cite:p:`hnb`).
The relevance is calculated by

.. math::

  Relevance(\text{feature}) = (P(\text{Class}  = 1 \mid \text{feature} = 1) - P (\text{Class}  = 1 \mid \text{feature} = 0))^2 \\
  +(P (\text{Class} = 0 | \text{feature} = 1) - P (\text{Class} = 0 | \text{feature} = 0))^2 
  

The Hierarchy Based Redundant Attribute Removal Naive Bayes Classifier (HNB) contains two phases which are executed for each test instance.
First, it considers every feature :math:`f_i`. If its value is 1 it removes all ancestors :math:`f_a` 
(:math:`f_a` is an ancestor of :math:`f_i`  iff :math:`f_i`  is reachable from :math:`f_a`), whose relevance is lower or equal than the relevance of :math:`f_i`.
Else if the value of :math:`f_i` is 0 it removes all descendants :math:`f_d` (reachable nodes from :math:`f_i`) whose relevance is lower.
In a second step it selects the :math:`k` most relevant features from the obtained set.
On the selected features of the training set an estimator can now calculate a prediction for the testing instance. 
Wan, Freitas and de Magalhaes suggest the usage of an Naive Bayes Classifier, the library allows the combination with any sklearn-compatible binary classifier.


The two phases can be executed separately from each other. 
The so called HNB-s selects all non-redundant features without performing any after-selecting steps.
In contrast the Relevance-based Naive Bayes (RNB) only selects the top-k-ranked features in descending order of their individual predictive power measured by their relevance (see above). 
