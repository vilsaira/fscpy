import numpy as np
import pytest
from numpy.testing import assert_allclose

from fsc import graph_from_edges


def test_laplacian_weights_symmetry_and_row_sum():
    graph = graph_from_edges(3, [0, 0, 1], [1, 1, 2], [2, 3, 4])
    expected = [[5, -5, 0], [-5, 9, -4], [0, -4, 4]]
    assert_allclose(graph.laplacian().toarray(), expected)
    assert_allclose(graph.laplacian().sum(axis=1), 0)


def test_negative_conductance_is_not_accepted():
    with pytest.raises(ValueError, match="conductances"):
        graph_from_edges(2, [0], [1], [-1])
