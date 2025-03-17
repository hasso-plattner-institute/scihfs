####################
Introduction
####################

With the growth of datasets nowadays, feature selection gains more importance in the area of machine learning and 
forms a typical pre-processing step in machine learning. Those methods aim to improve the predictive performance :cite:p:`fs`, 
for example by avoiding overfitting caused by noisy features. Feature selection thus may help to improve generalization.
Further a smaller training set can accelerate the training time, as the data the model has to include in its calculations is less.
Feature selection methods can also increase the explainability of methods since users can concentrate on the most important variables 
and their meaning for the model.

Many real-world settings contain hierarchical relations. In text mining, words can be ordered in 
generalization-specialization relationships :cite:p:`ristoski2014feature`, while in Bioinformatics the function 
of genes is often described as hierarchy as for example in the go-term ontology, which collects terms in a directed acyclic graph.
More information about the ontology can be found in :cite:t:`go`.

While there are implementations of hierarchical classification methods, for example in the Hiclass library by :cite:authors:`Hiclass`, 
there are not many implementations of feature selection methods that deal with hierarchical structures. Some hierarchical 
methods are implemented in the `kgextension library <https://github.com/om-hb/kgextension> [1]_, however it does not include an extensive collection.
Most methods assume flat feature spaces without hierarchical dependencies. But as redundancy and relevance of features 
can be obtained out of hierarchical information more precise :cite:p:`wanbook`, hierarchical feature selection aims at 
adding more information to improve the step of selection. 
Hence, we propose a library, that implements methods presented by Wan and Freitas (amongst others for example 
Hierarchy Based Redundant Attribute Removal, see :cite:p:`hnb`), Ristoski and Paulheim :cite:`ristoski2014feature`, Oudah and Henschel :cite:`oudah2018taxonomy`, Wang et al. :cite:`wang2002learning`, Jeong and Myaeng :cite:`jeong2013feature`
and Lu et al. :cite:`lu2013domain`.

.. [1] Knowledge Graph Extension for Python - https://github.com/om-hb/kgextension

References
-----------

 .. bibliography:: refs.bib