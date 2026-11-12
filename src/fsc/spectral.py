"""Spectral decomposition of FSC solutions in structural Laplacian eigenmodes.

For a structural Laplacian L with eigenpairs L v_k = lambda_k v_k, an FSC
potential solution phi has modal coefficients c_k = v_k^T phi and modal
energies E_k = lambda_k c_k^2. The energies sum to the energy dissipated in
the structural edges, phi^T L phi = I^T G^-1 I.
"""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from .parcellation.connectivity import _joint_solution, graph_from_sc
from .solver import fc_pairs


@dataclass
class SpectralResult:
    """Modal decomposition of parcellation FSC solutions.

    eigenvalues : (n_modes,)            structural Laplacian eigenvalues, ascending
    eigenvectors : (n_nodes, n_modes)   corresponding eigenvectors
    pairs : (n_pairs, 2)                FC pairs (pairwise: one row of results each;
                                        joint: all constrained pairs, one result row)
    coefficients : (n_rows, n_modes)    modal coefficients c_k of each solution
    energy : (n_rows, n_modes)          modal energies lambda_k c_k^2
    captured : (n_rows,)                share of the total energy in the kept modes
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    pairs: np.ndarray
    coefficients: np.ndarray
    energy: np.ndarray
    captured: np.ndarray


def laplacian_modes(laplacian):
    """Eigenvalues and eigenvectors of a graph Laplacian, without zero modes.

    One zero mode per connected component (isolated nodes included) is
    removed. Uses a dense eigendecomposition, so it is intended for
    parcellation-sized graphs.
    """
    dense = laplacian.toarray() if sparse.issparse(laplacian) else np.asarray(laplacian)
    n_components, _ = connected_components(sparse.csr_matrix(dense), directed=False)
    eigenvalues, eigenvectors = np.linalg.eigh(dense)
    return eigenvalues[n_components:], eigenvectors[:, n_components:]


def decompose(potentials, eigenvalues, eigenvectors):
    """Modal coefficients c_k = v_k^T phi and energies E_k = lambda_k c_k^2."""
    coefficients = eigenvectors.T @ np.asarray(potentials, dtype=float)
    return coefficients, eigenvalues * coefficients**2


def spectral_from_parcellations(sc, fc, *, method="pairwise", n_modes=None,
                                solver="auto"):
    """Spectral decomposition of parcellation FSC in structural eigenmodes.

    method="pairwise" decomposes the solution of every positive FC pair with
    a structural path, in a basis shared by all pairs. method="joint"
    decomposes the single joint solution (cycle-free FC pairs only).

    n_modes keeps the n_modes lowest modes. The total energy, and so the
    captured fraction, is always computed from the full spectrum.
    """
    graph = graph_from_sc(sc)
    if np.shape(fc) != (graph.n_nodes, graph.n_nodes):
        raise ValueError("SC and FC dimensions differ.")
    if n_modes is not None and int(n_modes) < 1:
        raise ValueError("n_modes must be positive.")

    laplacian = graph.laplacian()
    eigenvalues, eigenvectors = laplacian_modes(laplacian)

    if method == "pairwise":
        pairs, values = fc_pairs(fc)
        _, components = connected_components(laplacian, directed=False)
        connected = components[pairs[:, 0]] == components[pairs[:, 1]]
        pairs, values = pairs[connected], values[connected]

        # phi = (d / R) L^+ (e_i - e_j), written in the eigenbasis.
        delta = eigenvectors[pairs[:, 0]] - eigenvectors[pairs[:, 1]]
        resistance = (delta**2 / eigenvalues).sum(axis=1)
        coefficients = (values / resistance)[:, None] * delta / eigenvalues
    elif method == "joint":
        _, result = _joint_solution(sc, fc, solver)
        pairs, _ = fc_pairs(fc)
        coefficients, _ = decompose(result.potentials, eigenvalues, eigenvectors)
        coefficients = coefficients[None, :]
    else:
        raise ValueError(f"Unknown method: {method}")

    energy = eigenvalues * coefficients**2
    total = energy.sum(axis=1)

    kept = slice(None) if n_modes is None else slice(0, int(n_modes))
    energy_kept = energy[:, kept]
    with np.errstate(invalid="ignore", divide="ignore"):
        captured = energy_kept.sum(axis=1) / total

    return SpectralResult(
        eigenvalues=eigenvalues[kept],
        eigenvectors=eigenvectors[:, kept],
        pairs=pairs,
        coefficients=coefficients[:, kept],
        energy=energy_kept,
        captured=captured,
    )
