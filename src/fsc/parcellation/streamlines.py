"""Calculate individual streamline currents from FSC node potentials. Additionally, construct structural connectivity matrix from streamline assignments and resistances.

REFACTORED"""

import numpy as np

def _valid_streamlines(assignments, resistances, n_nodes):
    """Select valid streamlines and convert parcel labels to zero-based indices.

    assignments contains two parcel labels per streamline, numbered from 1.
    Label 0 indicates an unassigned endpoint. Resistances must be finite
    and positive.

    Returns a Boolean mask, valid parcel pairs, and their resistances.
    """
    endpoints = np.asarray(assignments, dtype=int)
    resistances = np.asarray(resistances, dtype=float)

    valid_parcels = np.all(
        (endpoints >= 1) & (endpoints <= n_nodes), axis=1
    )
    valid_resistances = np.isfinite(resistances) & (resistances > 0)

    valid = valid_parcels & valid_resistances

    return valid, endpoints[valid] - 1, resistances[valid]

def streamline_currents(potentials, assignments, resistances):
    """Calculate individual streamline currents from FSC node potentials.

    Each streamline current is the potential difference between its
    assigned parcels divided by its resistance.

    Currents are signed according to the ordering of the parcel endpoints.
    Invalid or unassigned streamlines receive zero current.

    Returns one current per input streamline, preserving their order.
    """
    potentials = np.asarray(potentials)
    valid, endpoints, resistance = _valid_streamlines(
        assignments, resistances, len(potentials)
    )
    source = endpoints[:, 0]
    target = endpoints[:, 1]
    currents = np.zeros(len(resistances))
    currents[valid] = (potentials[source] - potentials[target]) / resistance
    return currents

def sc_from_streamlines(assignments, resistances, n_nodes):
    """Construct an SC matrix by summing parallel streamline conductances.

    Each streamline connects two parcels and contributes 1/resistance
    to their structural connectivity. The resulting matrix is symmetric.

    Note about this function: FSC can take SC input from any other source too, using this function is not necessary. It is provided here for convenience, but it is not part of the FSC algorithm itself.
    """
    _, endpoints, resistance = _valid_streamlines(
        assignments, resistances, n_nodes
    )
    source = endpoints[:, 0]
    target = endpoints[:, 1]
    conductance = 1.0 / resistance
    sc = np.zeros((n_nodes, n_nodes))
    np.add.at(sc, (source, target), conductance)
    np.add.at(sc, (target, source), conductance)
    np.fill_diagonal(sc, 0)
    return sc