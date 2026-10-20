"""A five-node FSC example with an analytically known solution."""

import numpy as np

import fsc

# Structural edge weights are conductances.
sc = np.array([
    [0, 1, 1, 0, 0],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 1, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0],
], dtype=float)

# Impose a unit potential difference between parcels 1 and 4 (zero-based).
fc = np.zeros((5, 5))
fc[1, 4] = fc[4, 1] = 1.0

support = fsc.solve_from_parcellations(sc, fc)

print("SC:\n", sc)
print("FC:\n", fc)
print("FSC edge support:\n", np.round(support, 6))

# Three unit-resistance edges form the only path between parcels 1 and 4.
assert np.isclose(support[1, 0], 1 / 3)
assert np.isclose(support[0, 2], 1 / 3)
assert np.isclose(support[2, 4], 1 / 3)
assert np.isclose(support[2, 3], 0)
