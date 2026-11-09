"""Streamline-level FSC: distribute support over streamlines, compute
streamline currents from node potentials, and build SC from streamlines."""

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

def streamline_support(support, assignments, weights=None):
    """Distribute parcellation FSC support over individual streamlines.

    The support of each parcel pair is split among the streamlines
    connecting that pair, in proportion to their weights:

        value_s = support[a, b] * w_s / sum(w_t for streamlines t between a and b)

    The values of each pair's streamlines therefore sum to support[a, b],
    whatever method produced the SC matrix. Weights
    defaults to equal weights (streamline counts). If needed, pass the per-streamline weights the SC was built from, e.g. SIFT2 weights or 1 / length. Only relative weights within a pair matter.

    Assignments holds two parcel labels per streamline. Numbered from 1 in
    SC matrix order, with 0 for an unassigned endpoint (as in MRtrix3
    tck2connectome -out_assignments). Returns one non-negative value per
    input streamline in input order. Invalid, unassigned and
    self-connecting streamlines receive zero. Support on parcel pairs
    without any valid streamline is not mapped.
    """
    support = np.asarray(support, dtype=float)
    n_nodes = len(support)
    if support.shape != (n_nodes, n_nodes):
        raise ValueError("Support must be a square matrix.")

    assignments = np.asarray(assignments, dtype=int)
    if assignments.ndim != 2 or assignments.shape[1] != 2:
        raise ValueError("Assignments must hold two parcel labels per streamline.")

    if weights is None:
        weights = np.ones(len(assignments))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(assignments),):
        raise ValueError("One weight per streamline is required.")

    # The same validity rule as for resistances: finite and positive.
    valid, endpoints, w = _valid_streamlines(assignments, weights, n_nodes)
    low = np.minimum(endpoints[:, 0], endpoints[:, 1])
    high = np.maximum(endpoints[:, 0], endpoints[:, 1])

    pair_weight = np.zeros((n_nodes, n_nodes))
    np.add.at(pair_weight, (low, high), w)

    shares = np.zeros(len(low))
    distinct = low != high
    shares[distinct] = (
        support[low[distinct], high[distinct]]
        * w[distinct] / pair_weight[low[distinct], high[distinct]]
    )

    values = np.zeros(len(assignments))
    values[valid] = shares
    return values



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