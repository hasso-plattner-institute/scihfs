#####################
Preprocessing
#####################

.. currentmodule:: scihfs

The **HierarchicalPreprocessor** is responsible for preparing hierarchical datasets for feature selection. This preprocessor ensures that:

- Each node in the hierarchy corresponds to a column in the dataset.
- The dataset follows a **0-1 propagation rule**, meaning if a feature is `1`, all its descendants in the hierarchy must also be `1`.
- Missing hierarchy nodes are added, and unnecessary nodes are removed to maintain consistency.


Classes & Functions
---------------------------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   HierarchicalPreprocessor


Usage
---------------------------

To use the **HierarchicalPreprocessor**, initialize it with the hierarchy (as an adjacency matrix) and apply it to your dataset.

**Example:**

.. code-block:: python

   import networkx as nx
   import numpy as np

   from scihfs.helpers import get_columns_for_numpy_hierarchy
   from scihfs import HierarchicalPreprocessor

   # Example hierarchy as adjacency matrix
   edges = [(4, 5), (0, 1), (0, 3), (0, 4)]
   hierarchy = nx.DiGraph(edges)
   columns = get_columns_for_numpy_hierarchy(hierarchy, X.shape[1])
   hierarchy = nx.to_numpy_array(hierarchy)

   # Sample dataset
   X = np.array([
       [1, 0, 0],
       [0, 1, 1],
       [1, 1, 0]
   ])

   # Initialize and fit the preprocessor
   preprocessor = HierarchicalPreprocessor(hierarchy=hierarchy)
   preprocessor.fit(X, columns=columns)

   # Transform the dataset
   X_transformed = preprocessor.transform(X)

.. note::
   When ``X`` is a pandas ``DataFrame`` and the hierarchy is a named
   ``networkx.DiGraph``, the column-to-node mapping is derived automatically
   from the column names, so the ``columns`` argument of ``fit`` can be
   omitted. Columns without a matching node are added to the hierarchy under
   the virtual root (with a warning). See the eager learning example for an
   end-to-end DataFrame workflow.


Preprocessing Function
---------------------------

The following function demonstrates how to use `HierarchicalPreprocessor` to preprocess the hierarchy and dataset before performing feature selection:

.. code-block:: python

   # Preprocess hierarchy and dataset before feature selection to ensure all nodes
   # in the hierarchy have a corresponding node in the dataset and the other way around.
   def preprocess_data(hierarchy, X_train, X_test, columns):
       preprocessor = HierarchicalPreprocessor(hierarchy)
       preprocessor.fit(X_train, columns=columns)
       X_train_transformed = preprocessor.transform(X_train)
       X_test_transformed = preprocessor.transform(X_test)
       hierarchy_updated = preprocessor.to_adjacency_matrix()
       columns_updated = preprocessor.get_columns()
       return X_train_transformed, X_test_transformed, hierarchy_updated, columns_updated

.. note::
   ``to_adjacency_matrix()`` returns a ``scipy.sparse`` CSR array **by default**
   (``sparse=True``). The selectors accept a sparse hierarchy directly, so the
   updated hierarchy can be fed straight back into a selector without
   densifying -- which at large scale avoids allocating a dense adjacency matrix.
   Pass ``sparse=False`` if you specifically need a dense ``np.ndarray``.

   In either way: The result is dtype ``bool`` -- (a) because the hierarchy
   is purely structural, and (b) because it has the smallest memory footprint.

Handling Node Name Changes
---------------------------

Why Node Names Change
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a hierarchy graph is converted into an adjacency matrix, node names are replaced with numerical indices. This process can cause a loss of the original node labels, making it difficult to maintain a clear mapping between dataset columns and hierarchy nodes. To resolve this, the `_columns` attribute stores the mapping between the original node names and their transformed indices.

This ensures that, even after transformation, the preprocessor can correctly align the dataset columns with the appropriate hierarchical structure.

Example of Node Name Mapping
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Consider a simple hierarchy with nodes labeled `[4, 5, 0, 1, 3]`. When transformed into a NumPy adjacency matrix, these nodes will be replaced with indices `[0, 1, 2, 3, 4]`. The `_columns` attribute will track this mapping to ensure consistency in subsequent transformations.

.. code-block:: python

   original_nodes = [4, 5, 0, 1, 3]
   transformed_indices = [0, 1, 2, 3, 4]
   column_mapping = [2, 3, -1, 4]  # Example mapping

After fitting, the preprocessor adjusts these mappings so that the correct relationships between dataset columns and hierarchy nodes are preserved. This process ensures that feature selection algorithms operate on the intended hierarchical structure.

Node Names Are Compared as Strings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Node names of an ``nx.DiGraph`` hierarchy may be of any hashable type, but scihfs
only ever handles them as strings: scikit-learn stores DataFrame column labels as
``str`` in ``feature_names_in_``, so the mapping is derived by comparing each
``str(node)`` from the digraph against those labels, and
``get_feature_names_out()`` reports them in the same way. This is what lets an
integer-named hierarchy match a DataFrame whose column labels are the string
versions of those integers.

The flip side is that node names must be unique **as strings**. A hierarchy holding
both ``1`` and ``"1"`` has two distinct nodes but only one name to address them by,
so one of them could never be reached from a DataFrame column. Such a hierarchy is
rejected as soon as the mapping is derived from column names. Passing the
``columns`` mapping explicitly bypasses the name matching entirely and stays
allowed.


Implementation Details
----------------------

Internally, the preprocessor performs the following steps:

1. **Hierarchy Validation (`_check_dag`)**
   - Ensures that the hierarchy is a **Directed Acyclic Graph (DAG)**.

2. **Extending the Hierarchy (`_extend_dag`)**
   - Adds missing nodes for features without corresponding hierarchy entries.

3. **Pruning the Hierarchy (`_shrink_dag`)**
   - Removes nodes that do not correspond to any dataset columns.

4. **Ensuring Feature Propagation (`_propagate_ones`)**
   - Modifies the dataset so that if a feature is active (`1`), all its ancestors are also `1`.
