"""FSC on the example HCP connectivity matrices bundled with ENIGMA Toolbox.

Requires ENIGMA Toolbox and matplotlib; see the project README.
"""

import matplotlib.pyplot as plt
import numpy as np
from enigmatoolbox.datasets import load_fc, load_sc

import fsc

sc, sc_labels, _, _ = load_sc()
fc, fc_labels, _, _ = load_fc()

if sc.shape != fc.shape or not np.array_equal(sc_labels, fc_labels):
    raise ValueError("ENIGMA SC and FC parcellations do not match.")

fc[fc < 0] = 0

support = fsc.solve_from_parcellations(sc, fc)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for axis, matrix, title, cmap in zip(
    axes,
    (sc, fc, support),
    ("Structural connectivity", "Functional connectivity", "FSC edge support"),
    ("Blues", "Reds", "viridis"),
):
    im = axis.imshow(matrix, cmap=cmap, interpolation="nearest")
    axis.set_title(title)
    axis.set_xlabel("Parcel")
    axis.set_ylabel("Parcel")
    fig.colorbar(im, ax=axis, fraction=0.046, pad=0.04)

fig.tight_layout()
plt.show()
