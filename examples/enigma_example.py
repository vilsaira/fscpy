#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Minimal FSC example using ENIGMA Toolbox data.

This script:
1. Loads functional (FC) and structural (SC) connectivity matrices
2. Runs the FSC model
3. Prints basic outputs
4. Optionally visualizes the matrices
"""

import numpy as np
from enigmatoolbox.datasets import load_sc, load_fc
from fsc import FSC

def main():
    # ------------------------------------------------------------------
    # 1. Load ENIGMA connectivity data
    # ------------------------------------------------------------------
    fc_ctx, fc_labels, _, _ = load_fc()
    sc_ctx, sc_labels, _, _ = load_sc()

    # Ensure numpy arrays
    fc_ctx = np.asarray(fc_ctx, dtype=float)
    sc_ctx = np.asarray(sc_ctx, dtype=float)

    print("Loaded data:")
    print("FC shape:", fc_ctx.shape)
    print("SC shape:", sc_ctx.shape)

    # ------------------------------------------------------------------
    # 2. Run FSC model
    # ------------------------------------------------------------------
    model = FSC(FC=fc_ctx, SC=sc_ctx)

    phi = model.get_nodal_potentials()
    edge_currents = model.get_edge_currents()

    # ------------------------------------------------------------------
    # 3. Print summary
    # ------------------------------------------------------------------
    print("\nResults:")
    print("Nodal potentials shape:", phi.shape)
    print("Edge currents shape:", edge_currents.shape)
    print("Max current:", np.max(edge_currents))
    print("Min current:", np.min(edge_currents))

    # ------------------------------------------------------------------
    # 4. Optional visualization (requires nilearn)
    # ------------------------------------------------------------------
    try:
        from nilearn import plotting

        print("\nPlotting matrices...")

        plotting.plot_matrix(
            fc_ctx,
            figure=(8, 8),
            labels=fc_labels,
            cmap="Reds",
            title="Functional Connectivity (FC)",
        )

        plotting.plot_matrix(
            sc_ctx,
            figure=(8, 8),
            labels=sc_labels,
            cmap="Blues",
            title="Structural Connectivity (SC)",
        )

        plotting.plot_matrix(
            edge_currents,
            figure=(8, 8),
            labels=sc_labels,
            cmap="Greens",
            title="FSC Edge Currents",
        )

        plotting.show()

    except ImportError:
        print("\nOptional plotting skipped (install nilearn to enable).")


if __name__ == "__main__":
    main()