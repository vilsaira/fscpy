# FSC
Function-Structure Coupling (FSC)

This script contains all functions needed for the FSC analysis[1].

Author: Viljami Sairanen

[1] Sairanen, Viljami. ‘From nodes to pathways: an edge-centric model of brain structure-function coupling via constrained Laplacians’. https://doi.org/10.1101/2024.03.03.583186.

# A Python example of usage with the Human Connectome Project from the ENIGMA TOOLBOX.

```
from enigmatoolbox.datasets import load_sc, load_fc
from nilearn import plotting
from fsc import FSC

# Load cortico-cortical functional connectivity data
fc_ctx, fc_ctx_labels, _, _ = load_fc() 
# Load cortico-cortical structural connectivity data
sc_ctx, sc_ctx_labels, _, _ = load_sc() 
# Calculate cortico-cortical 'functio-structural current' connectivity data
fscc_ctx = FSC(V=fc_ctx, R=sc_ctx).get_nodal_currents_I()

# Plot cortico-cortical connectivity matrices
fc_plot = plotting.plot_matrix(fc_ctx, figure=(9, 9), labels=fc_ctx_labels, vmax=0.8, vmin=0, cmap='Reds')

sc_plot = plotting.plot_matrix(sc_ctx, figure=(9, 9), labels=sc_ctx_labels, vmax=10, vmin=0, cmap='Blues')

fsc_plot = plotting.plot_matrix(fscc_ctx, figure=(9, 9), labels=sc_ctx_labels, vmax=fscc_ctx.max(), vmin=fscc_ctx.min(), cmap='Greens')
```
