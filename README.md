# FSC
Function-Structure Coupling (FSC)

This script contains all functions needed for the FSC analysis[1].

Author: Viljami Sairanen

[1] Sairanen, Viljami. ‘From nodes to pathways: an edge-centric model of brain structure-function coupling via constrained Laplacians’. https://doi.org/10.1101/2024.03.03.583186.

# A Python example of usage with the Human Connectome Project from the ENIGMA TOOLBOX.

```
from enigmatoolbox.datasets import load_sc, load_fc
from fsc import FSC

fc_ctx, _, _, _ = load_fc()
sc_ctx, _, _, _ = load_sc()

fsc = FSC(FC=fc_ctx, SC=sc_ctx)
fscc_ctx = fsc.get_edge_currents()
```
