"""Tests for distributing parcellation FSC support over streamlines."""

import numpy as np
import pytest

import fsc
from fsc.parcellation.streamlines import sc_from_streamlines, streamline_support


def test_streamline_support_hand_computed():
    # Parcels 1-2 are joined by two streamlines (r = 1 and 0.5, total
    # conductance 3) and parcels 2-3 by one (r = 1). The second streamline
    # lists its endpoints in reverse order. The last three are invalid:
    # unassigned endpoint, self-connection, zero resistance.
    assignments = np.array([[1, 2], [2, 1], [2, 3], [0, 3], [2, 2], [1, 2]])
    resistances = np.array([1.0, 0.5, 1.0, 1.0, 1.0, 0.0])

    sc = sc_from_streamlines(assignments, resistances, 3)
    np.testing.assert_allclose(sc, [[0, 3, 0], [3, 0, 1], [0, 1, 0]])

    fc = np.zeros((3, 3))
    fc[0, 2] = fc[2, 0] = 1.0
    support = fsc.solve_from_parcellations(sc, fc)

    # Series path: R = 1/3 + 1 = 4/3, so the current is 0.75 on both edges.
    # Edge 1-2 splits 0.75 by conductance 1:2 -> 0.25 and 0.5.
    values = streamline_support(support, sc, assignments, resistances)
    np.testing.assert_allclose(values, [0.25, 0.5, 0.75, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("method", ["pairwise", "joint"])
def test_streamline_support_sums_to_edge_support(method):
    rng = np.random.default_rng(0)
    n = 6
    assignments = rng.integers(1, n + 1, size=(200, 2))
    resistances = rng.uniform(0.5, 2.0, size=200)
    sc = sc_from_streamlines(assignments, resistances, n)

    fc = np.zeros((n, n))
    if method == "pairwise":
        fc = rng.uniform(0.1, 0.6, (n, n))
        fc = np.triu(fc, 1) + np.triu(fc, 1).T
    else:
        fc[0, 5] = fc[5, 0] = 0.8   # two disjoint pairs: no cycle
        fc[1, 4] = fc[4, 1] = 0.5

    support = fsc.solve_from_parcellations(sc, fc, method=method)
    values = streamline_support(support, sc, assignments, resistances)

    # Summing the streamline values per parcel pair recovers the support.
    endpoints = assignments - 1
    total = np.zeros((n, n))
    np.add.at(total, (endpoints[:, 0], endpoints[:, 1]), values)
    total = total + total.T
    np.testing.assert_allclose(total, support, atol=1e-10)
    assert np.all(values >= 0)


def test_streamline_support_shape_check():
    with pytest.raises(ValueError):
        streamline_support(np.zeros((3, 3)), np.zeros((4, 4)), [[1, 2]], [1.0])
