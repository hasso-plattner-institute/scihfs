#######################
Lazy feature selection
#######################

.. currentmodule:: hfs

One main goal of the feature selection is the removal of hierarchical redundancy. The focus of the implemented methods in the area of lazy learning is binary classification. In this context redundancy is defined as features that are of same value (either 0 or 1) and hierarchically related, so they lie on the same path.
Several methods were proposed mainly by Wan and Freitas to handle this redundancy. We will shortly explain the algorithms and give insights in the implementation.


.. toctree::
    :includehidden:
    :maxdepth: 3

    hnb

.. toctree::
    :includehidden:
    :maxdepth: 3

    mrhip

.. toctree::
    :includehidden:
    :maxdepth: 3

    tan
