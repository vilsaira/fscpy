"""Spectral decomposition of FSC solutions in structural Laplacian eigenmodes.

For a structural Laplacian L with eigenpairs L v_k = lambda_k v_k, an FSC
potential solution phi has modal coefficients c_k = v_k^T phi and modal
energies E_k = lambda_k c_k^2. The energies sum to the energy dissipated in
the structural edges, phi^T L phi = I^T G^-1 I.
"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lobpcg

from .parcellation.connectivity import _joint_solution, graph_from_sc
from .solver import _voxel_pair_solutions, fc_pairs
from .weighted_graph import WeightedGraph

#--------------------------------------------------------------------------
# Parcellation workflow
#--------------------------------------------------------------------------

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

#--------------------------------------------------------------------------
# Voxel workflow
#--------------------------------------------------------------------------

FULL_BASIS_LIMIT = 5000      # largest graph for a dense full eigendecomposition


@dataclass
class VoxelModes:
    """Structural eigenmodes of the voxel white-matter graph.

    Computed from WM edges only (no supernode contacts), on the largest
    connected component, so the basis is shared by all FC pairs.

    eigenvalues : (n_modes,)              ascending, constant mode excluded
    eigenvectors : (n_voxels, n_modes)    rows = voxels of the main component
    wm_nodes : (n_voxels,)                graph node of each eigenvector row
    residuals : (n_modes,)                relative residual |L v - lambda v| / lambda
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    wm_nodes: np.ndarray
    residuals: np.ndarray


@dataclass
class VoxelSpectralResult:
    """Modal decomposition of voxel FSC pair solutions.

    pairs : (n_pairs, 2)                FC pairs with a structural path
    resistance : (n_pairs,)             effective resistance between the supernodes
    coefficients, energy : (n_pairs, n_modes)
    captured : (n_pairs,)               share of the main-component WM energy in the modes
    contact_share : (n_pairs,)          energy in supernode contact edges / total
    other_share : (n_pairs,)            energy in WM outside the main component / total

    The total energy of a pair is FC^2 / resistance.
    """

    eigenvalues: np.ndarray
    pairs: np.ndarray
    resistance: np.ndarray
    coefficients: np.ndarray
    energy: np.ndarray
    captured: np.ndarray
    contact_share: np.ndarray
    other_share: np.ndarray


def _wm_edges(graph, parcellation):
    """Supernode mask over nodes, and mask of edges between two WM voxels."""
    supernode = np.zeros(graph.n_nodes, dtype=bool)
    supernode[np.asarray(parcellation.roi_nodes)] = True
    return supernode, ~(supernode[graph.edge_i] | supernode[graph.edge_j])


def voxel_modes(graph, parcellation, n_modes=None, *, tol=1e-8, maxiter=500, seed=0):
    """Eigenmodes of the white-matter graph for voxel spectral analysis.

    n_modes=None computes the full basis with a dense eigendecomposition
    (small graphs only). An integer computes the n_modes lowest non-constant
    modes with LOBPCG and an algebraic multigrid preconditioner (requires
    pyamg). tol is LOBPCG's absolute residual tolerance; the relative
    residual of every mode is returned, and a warning is issued if any
    exceeds 1e-2.
    """
    supernode, wm_edge = _wm_edges(graph, parcellation)
    wm_all = np.flatnonzero(~supernode)
    position = np.full(graph.n_nodes, -1)
    position[wm_all] = np.arange(len(wm_all))
    laplacian = WeightedGraph(
        len(wm_all), position[graph.edge_i[wm_edge]], position[graph.edge_j[wm_edge]],
        graph.conductance[wm_edge],
    ).laplacian()

    _, components = connected_components(laplacian, directed=False)
    main = components == np.bincount(components).argmax()
    laplacian = laplacian[main][:, main].tocsr()
    wm_nodes = wm_all[main]
    n = laplacian.shape[0]

    if n_modes is None or n_modes >= n - 1:
        if n > FULL_BASIS_LIMIT:
            raise ValueError(
                f"A full basis for {n} voxels is too large; set n_modes."
            )
        eigenvalues, eigenvectors = laplacian_modes(laplacian)
    else:
        if n_modes < 1:
            raise ValueError("n_modes must be positive.")
        try:
            import pyamg
        except ImportError:
            raise ImportError(
                "voxel_modes with n_modes requires pyamg: pip install pyamg"
            ) from None

        shift = 1e-6 * laplacian.diagonal().mean()
        preconditioner = pyamg.smoothed_aggregation_solver(
            (laplacian + shift * sparse.identity(n, format="csr")).tocsr()
        ).aspreconditioner()
        start = np.random.default_rng(seed).normal(size=(n, int(n_modes)))
        constant = np.full((n, 1), 1.0 / np.sqrt(n))       # exclude the zero mode
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            eigenvalues, eigenvectors = lobpcg(
                laplacian, start, M=preconditioner, Y=constant,
                largest=False, tol=tol, maxiter=maxiter,
            )
        order = np.argsort(eigenvalues)
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]

    residuals = (
        np.linalg.norm(laplacian @ eigenvectors - eigenvectors * eigenvalues, axis=0)
        / eigenvalues
    )
    if residuals.size and residuals.max() > 1e-2:
        warnings.warn(
            f"{np.sum(residuals > 1e-2)} of {len(eigenvalues)} modes have a relative "
            f"residual above 1e-2 (largest {residuals.max():.1e}); increase maxiter "
            "or lower tol.",
            stacklevel=2,
        )
    return VoxelModes(eigenvalues, eigenvectors, wm_nodes, residuals)


def spectral_fsc_voxel(graph, parcellation, fc, modes, pairs=None, *,
                       solver="auto", rtol=1e-6, maxiter=None):
    """Spectral decomposition of voxel FSC pair solutions.

    Each pair is solved exactly as in solve_fsc_voxel() (only its own two
    supernodes take part), and the white-matter part of its potentials is
    projected onto the shared modes from voxel_modes().
    """
    _, wm_edge = _wm_edges(graph, parcellation)
    in_main = np.zeros(graph.n_nodes, dtype=bool)
    in_main[modes.wm_nodes] = True
    main_edge = wm_edge & in_main[graph.edge_i] & in_main[graph.edge_j]

    rows = {key: [] for key in ("pairs", "resistance", "coefficients", "energy",
                                "captured", "contact_share", "other_share")}
    for a, b, _, resistance, potentials, active in _voxel_pair_solutions(
        graph, parcellation, fc, pairs, solver=solver, rtol=rtol, maxiter=maxiter,
    ):
        drop = potentials[graph.edge_i] - potentials[graph.edge_j]
        edge_energy = np.where(active, graph.conductance * drop**2, 0.0)
        total = edge_energy.sum()
        wm_energy = edge_energy[wm_edge].sum()
        main_energy = edge_energy[main_edge].sum()

        coefficients, energy = decompose(
            potentials[modes.wm_nodes], modes.eigenvalues, modes.eigenvectors
        )
        rows["pairs"].append((a, b))
        rows["resistance"].append(resistance)
        rows["coefficients"].append(coefficients)
        rows["energy"].append(energy)
        rows["captured"].append(energy.sum() / main_energy)
        rows["contact_share"].append((total - wm_energy) / total)
        rows["other_share"].append((wm_energy - main_energy) / total)

    n_modes = len(modes.eigenvalues)
    return VoxelSpectralResult(
        eigenvalues=modes.eigenvalues,
        pairs=np.array(rows["pairs"], dtype=int).reshape(-1, 2),
        resistance=np.array(rows["resistance"]),
        coefficients=np.array(rows["coefficients"]).reshape(-1, n_modes),
        energy=np.array(rows["energy"]).reshape(-1, n_modes),
        captured=np.array(rows["captured"]),
        contact_share=np.array(rows["contact_share"]),
        other_share=np.array(rows["other_share"]),
    )
