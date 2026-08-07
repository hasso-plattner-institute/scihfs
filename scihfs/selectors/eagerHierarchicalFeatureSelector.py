"""
Base class for estimators for eager hierarchical feature selection.
"""

from abc import abstractmethod

import numpy as np
from sklearn.feature_selection import SelectorMixin

from scihfs.selectors import HierarchicalEstimator


class EagerHierarchicalFeatureSelector(SelectorMixin, HierarchicalEstimator):
    """Abstract base class for eager feature selectors using hierarchical data.

    Eager selectors are supervised: ``fit`` requires a target ``y``.
    Subclasses implement their selection algorithm in ``_select``, which
    runs after the single input-validation pass and the hierarchy setup
    and fills ``selected_features_`` with the names of all hierarchy nodes
    that are left after feature selection.

    A DataFrame passed to ``fit`` together with a named ``nx.DiGraph``
    hierarchy auto-derives the column->node mapping from the feature
    names -- in this specific (but encouraged) case the ``columns`` argument
    can be omitted. Feature names without a matching node raise ``ValueError``
    (unlike the ``HierarchicalPreprocessor``, a selector cannot extend
    the hierarchy).

    Eager selectors also support scikit-learn's output API (inherited from
    ``SelectorMixin``): ``get_feature_names_out`` returns the names of the
    selected features (the captured DataFrame column names, or ``x0``-style
    fallbacks for unnamed input), and ``set_output(transform="pandas")``
    makes ``transform`` return a DataFrame labelled with them.
    """

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def _fit(self, X, y):
        """Runs the eager feature selection on the validated X and y.

        Rejects a hierarchy/column mismatch, resets ``selected_features_`` and
        delegates to the subclasses' ``_select``.
        """
        self._reject_column_node_mismatch()

        # self.selected_features_ includes all node names for selected nodes.
        # self._columns maps them to the respective column in X.
        self.selected_features_ = []
        self._select(X, y)

    @abstractmethod
    def _select(self, X, y):
        """The subclass's feature selection algorithm.

        Called with the validated X and y after the hierarchy setup, and
        expected to fill ``self.selected_features_`` with the names of the
        selected hierarchy nodes.
        """

    def _get_support_mask(self):
        # Implements _get_support_mask method from SelectorMixin to
        # indicate the selected features from X.
        selected_indices = [self._column_index(node) for node in self.selected_features_]
        return np.asarray(
            [
                True if index in selected_indices else False
                for index in range(self.n_features_in_)
            ]
        )
