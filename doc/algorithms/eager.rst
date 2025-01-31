#######################
Eager Feature Selection
#######################

.. currentmodule:: hfs


All eager feature selection algorithms can be used in the same way. The feature selector object is initialized with
the hierarchy in the form of a numpy array and all needed hyperparameters. Then the fit method can be called which triggers
the feature selection algorithm. During the algorithm, the most relevant features are selected based on the hierarchy
information and certain relevancy metrics. The selected features are saved in the selector object. To transform the dataset
the transform method can be called on the selector after fitting it. This method outputs the dataset with only the selected features.

There are several hierarchical feature selection algorithms that select the features in different ways. They will be discussed in the following sections.


.. toctree::
    :includehidden:
    :maxdepth: 3

    tsel

.. toctree::
    :includehidden:
    :maxdepth: 3

    shsel

.. toctree::
    :includehidden:
    :maxdepth: 3

    gtd

.. toctree::
    :includehidden:
    :maxdepth: 3

    hill_climbing
