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

def streamline_support(support, sc, assignments, resistances):
    """Distribute parcellation FSC support over individual streamlines.

    Each streamline between parcels a and b receives

        support[a, b] / (resistance * sc[a, b]),

    the share of the edge support proportional to its conductance. This is
    the absolute current the streamline would carry as a resistor between
    the solved potentials, accumulated in the same way as the support
    matrix. It applies to both solve_from_parcellations() methods 'joint' and 'pairwise'.

    When sc was built from the same streamlines with sc_from_streamlines(),
    the values of the streamlines between each parcel pair sum to
    support[a, b]. With SC from another source, the values are still
    proportional to streamline conductance but need not sum to it.

    Returns one non-negative value per input streamline preserving their
    order. Invalid, unassigned and self-connecting streamlines, and
    streamlines whose parcel pair has no SC edge receive zero weight.
    """
    support = np.asarray(support, dtype=float)
    sc = np.asarray(sc, dtype=float)
    n_nodes = len(support)
    if support.shape != (n_nodes, n_nodes) or sc.shape != support.shape:
        raise ValueError("Support and SC must be square matrices of equal size.")

    valid, endpoints, resistance = _valid_streamlines(
        assignments, resistances, n_nodes
    )
    source = endpoints[:, 0]
    target = endpoints[:, 1]

    edge_sc = sc[source, target]
    has_edge = (source != target) & (edge_sc > 0)

    shares = np.zeros(len(source))
    shares[has_edge] = support[source[has_edge], target[has_edge]] / (
        resistance[has_edge] * edge_sc[has_edge]
    )

    values = np.zeros(len(np.asarray(resistances)))
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