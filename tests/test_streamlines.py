import numpy as np
from numpy.testing import assert_allclose

from fsc.parcellation.streamlines import (sc_from_streamlines, streamline_currents,
                                         current_guided_paths, streamline_currents_on_paths)


def test_assignments_conductance_and_current():
    assignments = [[1, 2], [1, 2], [2, 3], [0, 2]]
    resistance = [1, 2, 4, 1]
    sc = sc_from_streamlines(assignments, resistance, 3)
    assert_allclose(sc, [[0, 1.5, 0], [1.5, 0, .25], [0, .25, 0]])
    assert_allclose(streamline_currents([2, 1, 0], assignments, resistance),
                    [1, .5, .25, 0])


def test_current_guided_streamline_paths():
    sc = [[0, 1, 0], [1, 0, 1], [0, 1, 0]]
    fc = [[0, 0, 1], [0, 0, 0], [1, 0, 0]]
    current = [[0, 1, 0], [-1, 0, 1], [0, -1, 0]]
    paths = current_guided_paths(sc, fc, current)
    assert paths[0, 2] == [0, 1, 2]
    values = streamline_currents_on_paths([2, 1, 0], [[1, 2], [2, 3], [1, 3]],
                                           [1, 1, 1], paths)
    assert_allclose(values[0, 2], [1, 1, 0])
