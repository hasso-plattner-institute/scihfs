############################
Feature selection algorithms
############################

.. currentmodule:: hfs

Feature selection preprocesses the data by removing irrelevant features, that is features not correlated with the class variable,
and redundant features, which are strongly correlated to other features. Methods performing feature selection can be classified as
wrapper or filter methods. The wrapper approach evaluates the selected subset of features upon the predictive performance of an
afterwards-built classifier. In contrast, the filter approach selects the set of features based on a quality measurement independent
of the classification algorithm. So the set of features is selected before the actual prediction using measures like the relevance
of features. This library implements the filter approach and thus allows the usage of the feature selection with user-selected estimators.


Machine learning methods can be further categorized into lazy :cite:p:`lazy` and eager learning. Former builds one classification model for each testing instance, while the latter learns one classifier during the training phase and uses this one for predictions.
The library provides feature selection methods for both approaches, which we will explain in the following.

.. toctree::
    :includehidden:
    :maxdepth: 3

    eager

.. toctree::
    :includehidden:
    :maxdepth: 3

    lazy
