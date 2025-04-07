"""
Estimators for feature selection on hierarchical data.
"""

from scihfs._version import __version__
from scihfs.data_utils import create_mapping_columns_to_nodes
from scihfs.helpers import get_columns_for_numpy_hierarchy
from scihfs.preprocessing import HierarchicalPreprocessor
from scihfs.selectors import (
    HIP,
    HNB,
    MR,
    RNB,
    TAN,
    GreedyTopDownSelector,
    HNBs,
    SHSELSelector,
    TSELSelector,
)
from scihfs.selectors.eagerHierarchicalFeatureSelector import (
    EagerHierarchicalFeatureSelector,
    HierarchicalEstimator,
)
from scihfs.selectors.hill_climbing import (
    BottomUpSelector,
    HillClimbingSelector,
    TopDownSelector,
)
from scihfs.selectors.lazyHierarchicalFeatureSelector import (
    LazyHierarchicalFeatureSelector,
)

__all__ = [
    "TSELSelector",
    "SHSELSelector",
    "TopDownSelector",
    "BottomUpSelector",
    "HillClimbingSelector",
    "GreedyTopDownSelector",
    "HierarchicalEstimator",
    "EagerHierarchicalFeatureSelector",
    "HierarchicalPreprocessor",
    "LazyHierarchicalFeatureSelector",
    "HIP",
    "HNB",
    "HNBs",
    "MR",
    "RNB",
    "TAN",
    "get_columns_for_numpy_hierarchy",
    "create_mapping_columns_to_nodes",
    "__version__",
]
