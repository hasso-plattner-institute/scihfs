import pytest

from scihfs.data_utils import create_mapping_columns_to_nodes


@pytest.mark.parametrize(
    "hierarchy",
    [
        "hierarchy1",
        "hierarchy1_2",
        "hierarchy2",
        "hierarchy3",
    ],
)
def test_create_mapping_columns_to_nodes(hierarchy, dataframe, request):
    hierarchy = request.getfixturevalue(hierarchy)
    mapping = create_mapping_columns_to_nodes(dataframe, hierarchy)
    nodes = list(hierarchy.nodes)
    for index, node in enumerate(dataframe.columns):
        assert nodes[mapping[index]] == node
