# FSC (Function–Structure Coupling)

FSC is a Python implementation of the **Function–Structure Coupling** model, an edge-centric framework for analyzing brain connectivity using constrained Laplacians.

The method links **functional connectivity (FC)** and **structural connectivity (SC)** by solving a network flow model that explains functional interactions through structural pathways.

---

## Installation

Install FSC from PyPI:

```bash
pip install fscpy
```

Import in Python:

```python
from fsc import FSC
```

---

## Overview

FSC formulates function-structure coupling as a constrained network problem:

- Functional connectivity (FC) defines **pairwise constraints** (imposed potential differences)
- Structural connectivity (SC) defines the **network topology and weights**
- The model solves for nodal potentials (φ) and edge-level currents (I)

Mathematically:

```
I_ij = SC_ij * (φ_i - φ_j)
```

where:
- φ = nodal potentials  
- SC = structural connectivity (weights)  
- I = edge-level current (flow)

---

## Quick Example

```python
from enigmatoolbox.datasets import load_sc, load_fc
from fsc import FSC

# Load connectivity matrices
fc_ctx, _, _, _ = load_fc()
sc_ctx, _, _, _ = load_sc()

# Run FSC model
fsc = FSC(FC=fc_ctx, SC=sc_ctx)

# Get outputs
phi = fsc.get_nodal_potentials()
edge_currents = fsc.get_edge_currents()
```

---

## Outputs

Main methods:

- `get_nodal_potentials()`  
  → Nodal potentials (φ)

- `get_edge_currents()`  
  → Edge-level currents (I)

- `get_voltage_difference_matrix()`  
  → Pairwise potential differences (φ_i − φ_j)

- `get_graph_laplacian()`  
  → Structural graph Laplacian

---

## Examples

See the [`examples/`](examples/) directory for:

- ENIGMA Toolbox example
- Visualization scripts

Run an example:

```bash
python examples/enigma_example.py
```

---

## Optional Dependencies

Some examples require additional packages:

```bash
pip install enigmatoolbox nilearn
```

---

## Notes

- FC is interpreted as **imposed pairwise potential differences**
- SC is treated as a **weighted adjacency matrix (conductance)**  
- Only the upper triangle of FC is used to define constraints
- The model uses **Modified Nodal Analysis (MNA)**

---

## Applications

FSC can be used for:

- Studying function-structure coupling in brain networks  
- Identifying structural pathways supporting functional connectivity  
- Network flow analysis on connectomes  
- Tractography filtering and visualization  

---

## Reference

Sairanen, Viljami.  
**From nodes to pathways: an edge-centric model of brain function-structure coupling via constrained Laplacians**  
[https://doi.org/10.1101/2024.03.03.583186](https://www.biorxiv.org/content/10.1101/2024.03.03.583186v6)

---

## Author

Viljami Sairanen

---

## License

MIT License (see `LICENSE` file)
