"""Tests for distributing parcellation FSC support over streamlines."""

import numpy as np
import pytest

import fsc
from fsc.parcellation.streamlines import sc_from_streamlines, streamline_support


def test_streamline_support_hand_computed():
    assignments = np.array([[1, 2], [2, 1], [2, 3], [0, 3], [2, 2], [1, 2]])
    resistances = np.array([1.0, 0.5, 1.0, 1.0, 1.0, 0.0])
    weights = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 0.0])    # 1 / r; 0 is invalid

    sc = sc_from_streamlines(assignments, resistances, 3)
    fc = np.zeros((3, 3))
    fc[0, 2] = fc[2, 0] = 1.0
    support = fsc.solve_from_parcellations(sc, fc)

    values = streamline_support(support, assignments, weights)
    np.testing.assert_allclose(values, [0.25, 0.5, 0.75, 0, 0, 0], atol=1e-12)


@pytest.mark.parametrize("method", ["pairwise", "joint"])
def test_streamline_support_sums_to_edge_support(method):
    rng = np.random.default_rng(0)
    n = 6
    assignments = rng.integers(1, n + 1, size=(200, 2))
    weights = rng.uniform(0.5, 2.0, size=200)
    sc = sc_from_streamlines(assignments, 1 / weights, n)

    fc = np.zeros((n, n))
    if method == "pairwise":
        fc = rng.uniform(0.1, 0.6, (n, n))
        fc = np.triu(fc, 1) + np.triu(fc, 1).T
    else:
        fc[0, 5] = fc[5, 0] = 0.8
        fc[1, 4] = fc[4, 1] = 0.5

    support = fsc.solve_from_parcellations(sc, fc, method=method)
    values = streamline_support(support, assignments, weights)

    endpoints = assignments - 1
    total = np.zeros((n, n))
    np.add.at(total, (endpoints[:, 0], endpoints[:, 1]), values)
    np.testing.assert_allclose(total + total.T, support, atol=1e-10)
    assert np.all(values >= 0)


def test_streamline_support_external_sc():
    """SC from elsewhere (rescaled, normalised) still maps exactly."""
    rng = np.random.default_rng(1)
    n = 5
    assignments = rng.integers(1, n + 1, size=(100, 2))
    counts = sc_from_streamlines(assignments, np.ones(100), n)
    external_sc = counts / counts.max()       # e.g. max-normalised by another tool

    fc = rng.uniform(0.1, 0.6, (n, n))
    fc = np.triu(fc, 1) + np.triu(fc, 1).T
    support = fsc.solve_from_parcellations(external_sc, fc)

    values = streamline_support(support, assignments)      # default: counts
    total = np.zeros((n, n))
    np.add.at(total, (assignments[:, 0] - 1, assignments[:, 1] - 1), values)
    np.testing.assert_allclose(total + total.T, support, atol=1e-10)

    # Only relative weights within a pair matter.
    np.testing.assert_allclose(
        streamline_support(support, assignments, np.full(100, 7.0)), values
    )


def test_streamline_support_input_checks():
    with pytest.raises(ValueError):
        streamline_support(np.zeros((3, 4)), [[1, 2]])
    with pytest.raises(ValueError):
        streamline_support(np.zeros((3, 3)), [[1, 2, 3]])
    with pytest.raises(ValueError):
        streamline_support(np.zeros((3, 3)), [[1, 2]], weights=[1.0, 2.0])
