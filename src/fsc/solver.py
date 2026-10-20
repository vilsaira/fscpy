"""Function-structure coupling through a constrained graph Laplacian."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, minres, spsolve, cg, LinearOperator
from scipy.sparse.csgraph import connected_components

@dataclass
class FunctionalConnectivity:
    """Functional connectivity expressed as potential constraints.

    incidence: Constraint incidence matrix B (n_nodes, n_constraints).
    constraints: Imposed potential differences s_fc (n_constraints,).
    """
    incidence: sparse.csr_matrix
    constraints: np.ndarray


@dataclass
class FSCResult:
    potentials: np.ndarray
    edge_currents: np.ndarray
    source_currents: np.ndarray
    constraint_residual: np.ndarray
    current_balance: np.ndarray
    relative_residual: float

    def edge_current_matrix(self, structural_graph, absolute=True):
        """Return structural edge currents as a connectivity matrix.

        Small graphs return a dense array.
        Large graphs return a sparse matrix.
        I decided that the N for large is 2000 today.
        """
        N_is_large = 2000
        
        currents = self.edge_currents
        if absolute:
            currents = np.abs(currents)
        
        n_nodes = structural_graph.n_nodes

        matrix = sparse.coo_matrix(
            (currents, (structural_graph.edge_i, structural_graph.edge_j)),
            shape=(n_nodes, n_nodes)
        ).tocsr()

        if absolute:
            matrix += matrix.T
        else:
            matrix -= matrix.T

        if len(currents) < N_is_large and n_nodes <= N_is_large:
            return matrix.toarray()

        return matrix


def fc_pairs(fc):
    """Get finite, nonzero FC pairs from the upper triangle of a square matrix."""
    fc = np.asarray(fc, dtype=float)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError("FC must be square.")
    if not np.allclose(fc, fc.T, equal_nan=True):
        raise ValueError("FC must be symmetric.")

    selected = np.triu(np.isfinite(fc) & (fc != 0), k=1)
    pairs = np.column_stack(np.nonzero(selected))
    values = fc[pairs[:, 0], pairs[:, 1]]
    return pairs, values


def functional_connectivity_from_fc_pairs(pairs, values, n_nodes):
    """Construct FunctionalConnectivity from functional node pairs."""
    pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(pairs) != len(values):
        raise ValueError("FC pairs and values must have matching lengths.")
    if np.any(pairs < 0) or np.any(pairs >= n_nodes):
        raise ValueError("FC pair index outside graph.")

    n_pairs = len(pairs)

    rows = np.empty(2 * n_pairs, dtype=int)
    columns = np.empty(2 * n_pairs, dtype=int)
    coefficients = np.empty(2 * n_pairs, dtype=float)

    rows[0::2] = pairs[:, 0]
    rows[1::2] = pairs[:, 1]

    columns[0::2] = np.arange(n_pairs)
    columns[1::2] = np.arange(n_pairs)

    coefficients[0::2] = 1.0
    coefficients[1::2] = -1.0

    incidence = sparse.csr_matrix(
        (coefficients, (rows, columns)),
        shape=(n_nodes, n_pairs)
    )

    return FunctionalConnectivity(incidence, values)


def functional_connectivity_from_fc_matrix(fc):
    """Construct FunctionalConnectivity from an FC matrix."""
    pairs, values = fc_pairs(fc)

    return functional_connectivity_from_fc_pairs(pairs, values, len(fc))



def _prepare_mna(structural_graph, ground_node=0):
    n_nodes = structural_graph.n_nodes

    if not 0 <= ground_node < n_nodes:
        raise ValueError("Invalid ground node.")

    laplacian = structural_graph.laplacian()

    _, components = connected_components(
        laplacian, directed=False
    )

    _, representatives = np.unique(
        components, return_index=True
    )

    return laplacian, components, representatives

def _mna_mask(incidence, components, representatives, ground_node):
    """Ground one node per component joined by structural or FC connections."""
    n_components = len(representatives)

    if n_components == 1:
        mask = np.ones(len(components), dtype=bool)
        mask[ground_node] = False
        return mask

    # Connect structural components through functional constraints.
    entries = incidence.tocoo()

    links = sparse.coo_matrix(
        (
            np.ones(len(entries.row)),
            (components[entries.row], entries.col),
        ),
        shape=(n_components, incidence.shape[1]),
    ).tocsr()

    adjacency = sparse.bmat([
        [None, links],
        [links.T, None],
    ], format="csr")

    _, groups = connected_components(adjacency, directed=False)
    structural_groups = groups[:n_components]

    # Select one ground node per resulting group.
    _, first = np.unique(structural_groups, return_index=True)
    ground_nodes = representatives[first].copy()

    # Preserve the requested ground_node in its group.
    selected_group = structural_groups[components[ground_node]]
    ground_nodes[structural_groups[first] == selected_group] = ground_node

    mask = np.ones(len(components), dtype=bool)
    mask[ground_nodes] = False

    return mask



def _solve_mna(structural_graph, functional_connectivity, prepared, *,
               ground_node=0, solver="auto", rtol=1e-8, maxiter=None):
    """Solve FSC using modified nodal analysis.

    Solve [L  B; B.T  0] [potential; source_current] = [0; |FC|].
    ``functional_connectivity`` contains both B and its target values.

    ``solver='auto'`` uses sparse direct solving for small systems and
    MINRES for larger ones. Inaccurate convergence raises an error.
    The caller is responsible for providing mutually compatible constraints.
    """
    n_nodes = structural_graph.n_nodes
    incidence = functional_connectivity.incidence.tocsr()
    constraints = np.abs(np.asarray(functional_connectivity.constraints, dtype=float).reshape(-1))
    n_constraints = len(constraints)

    if incidence.shape != (n_nodes, n_constraints):
        raise ValueError("Functional constraint dimensions do not match the graph.")
    if not np.all(np.isfinite(constraints)):
        raise ValueError("Functional constraint values must be finite.")
    if solver not in ("auto", "direct", "minres"):
        raise ValueError(f"Unknown solver: {solver}")

    if n_constraints == 0:
        return FSCResult(
            np.zeros(n_nodes), np.zeros(len(structural_graph.edge_i)),
            np.empty(0), np.empty(0), np.zeros(n_nodes), 0.0,
        )

    laplacian, components, representatives = prepared

    mask = _mna_mask(
        incidence, components, representatives, ground_node
    )

    grounded_laplacian = laplacian[mask][:, mask].tocsr()

    reduced_incidence = incidence[mask]
    n_free = np.count_nonzero(mask)

    reduced_matrix = sparse.bmat([
        [grounded_laplacian, reduced_incidence],
        [reduced_incidence.T, None],
    ]).tocsr()

    reduced_rhs = np.concatenate((np.zeros(n_free), constraints))
    solution = np.zeros(n_free + n_constraints)

    if solver == "auto":
        solver = "direct" if len(reduced_rhs) <= 1000 else "minres"

    if solver == "direct":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution[:] = spsolve(reduced_matrix.tocsc(), reduced_rhs)
        except MatrixRankWarning as exc:
            raise ValueError("Singular MNA system: check functional constraints.") from exc
    else:
        solution[:], status = minres(
            reduced_matrix, reduced_rhs, rtol=rtol, maxiter=maxiter
        )
        if status != 0:
            raise RuntimeError(f"MINRES did not converge (status={status}).")

    potentials = np.zeros(n_nodes)
    potentials[mask] = solution[:n_free]
    source_currents = solution[n_free:]
    edge_currents = structural_graph.conductance * (
        potentials[structural_graph.edge_i] - potentials[structural_graph.edge_j]
    )
    constraint_residual = incidence.T @ potentials - constraints
    current_balance = laplacian @ potentials + incidence @ source_currents

    residual = np.linalg.norm(np.concatenate((current_balance, constraint_residual)))
    scale = np.linalg.norm(constraints)
    relative_residual = float(residual / scale) if scale else float(residual)

    if not np.isfinite(relative_residual) or relative_residual > max(10 * rtol, 1e-10):
        raise RuntimeError(
            f"MNA residual diagnostics:\n"
            f"  Current balance: {np.linalg.norm(current_balance):.6g}\n"
            f"  ROI equality constraints: {np.linalg.norm(constraint_residual[:-1]):.6g}\n"
            f"  FC potential constraint: {constraint_residual[-1]:.6g}\n"
            f"  Maximum constraint error: {np.max(np.abs(constraint_residual)):.6g}\n"
            f"  Potential range: {potentials.min():.6g}, {potentials.max():.6g}\n"
            f"  Relative residual: {relative_residual:.6g}"
        )


    return FSCResult(
        potentials, edge_currents, source_currents,
        constraint_residual, current_balance, relative_residual,
    )



def solve_fsc(structural_graph, functional_connectivity, *, ground_node=0,
              solver="auto", rtol=1e-8, maxiter=None):
    prepared = _prepare_mna(structural_graph, ground_node)
    return _solve_mna(
        structural_graph,
        functional_connectivity,
        prepared,
        solver=solver,
        rtol=rtol,
        maxiter=maxiter,
        ground_node=ground_node,
    )




def solve_fsc_voxel(structural_graph, parcellation, fc, pairs=None, *,
                    ground_node=0, solver="auto", rtol=1e-4, maxiter=None):
    """Compute voxelwise FSC from reusable ROI-relative Laplacian solutions.

    Each ROI is a structural graph supernode. Solving unit-current
    problems relative to a reference ROI provides basis potentials
    whose differences solve the pairwise MNA equations.
    """
    fc = np.asarray(fc, dtype=float)
    roi_nodes = parcellation.roi_nodes
    n_rois = len(roi_nodes)
    n_nodes = structural_graph.n_nodes

    if fc.shape != (n_rois, n_rois):
        raise ValueError("FC and parcellation dimensions differ.")

    if solver not in ("auto", "cg", "direct"):
        raise ValueError(f"Unknown voxel solver: {solver}")

    if pairs is None:
        pairs, values = fc_pairs(fc)
    else:
        pairs = np.asarray(pairs, dtype=int).reshape(-1, 2)

        if np.any(pairs < 0) or np.any(pairs >= n_rois):
            raise ValueError("FC pair index outside parcellation.")

        values = fc[pairs[:, 0], pairs[:, 1]]

    edge_currents = np.zeros(len(structural_graph.edge_i))

    # Ignore zero, nonfinite and self-connections.
    valid = (
        np.isfinite(values)
        & (values != 0)
        & (pairs[:, 0] != pairs[:, 1])
    )
    pairs = pairs[valid]
    values = values[valid]

    if len(pairs) == 0:
        return edge_currents

    # Prepare the grounded structural Laplacian.
    laplacian, components, representatives = _prepare_mna(
        structural_graph, ground_node
    )

    # Functional pairs without a structural path contribute no current.
    pair_nodes = roi_nodes[pairs]
    connected = (
        components[pair_nodes[:, 0]]
        == components[pair_nodes[:, 1]]
    )
    pairs = pairs[connected]
    values = values[connected]

    if len(pairs) == 0:
        return edge_currents

    ground_nodes = representatives.copy()
    ground_nodes[components[ground_node]] = ground_node

    mask = np.ones(n_nodes, dtype=bool)
    mask[ground_nodes] = False

    matrix = laplacian[mask][:, mask].tocsr()

    diagonal = matrix.diagonal()
    preconditioner = LinearOperator(
        matrix.shape,
        matvec=lambda x: x / diagonal,
        dtype=float,
    )

    # Choose one reference ROI per connected component.
    active_rois = np.unique(pairs)
    references = {}

    for roi in active_rois:
        component = components[roi_nodes[roi]]
        if component not in references:
            references[component] = roi

    # Compute reusable unit-current potential fields.
    basis = {}

    for roi in active_rois:
        component = components[roi_nodes[roi]]
        reference = references[component]

        if roi == reference:
            basis[roi] = np.zeros(n_nodes)
            continue

        rhs = np.zeros(n_nodes)
        rhs[roi_nodes[roi]] = 1.0
        rhs[roi_nodes[reference]] = -1.0

        if solver == "direct":
            solution = spsolve(matrix.tocsc(), rhs[mask])
        else:
            solution, status = cg(
                matrix,
                rhs[mask],
                M=preconditioner,
                rtol=rtol,
                maxiter=maxiter,
            )

            if status != 0:
                raise RuntimeError(
                    f"CG did not converge for ROI {roi} "
                    f"(status={status})."
                )

        potentials = np.zeros(n_nodes)
        potentials[mask] = solution
        basis[roi] = potentials

    # Reconstruct FC-pair solutions from differences of basis fields.
    for pair, value in zip(pairs, values):
        i, j = pair
        source, target = roi_nodes[pair]

        potentials = basis[i] - basis[j]
        resistance = potentials[source] - potentials[target]

        if not np.isfinite(resistance) or resistance <= 0:
            raise RuntimeError(
                f"Invalid effective resistance for ROI pair {pair}."
            )

        current = abs(value) / resistance
        potentials *= current

        # Check current conservation at every graph node.
        balance = laplacian @ potentials
        balance[source] -= current
        balance[target] += current

        relative_residual = np.linalg.norm(balance) / abs(value)

        if (
            not np.isfinite(relative_residual)
            or relative_residual > max(10 * rtol, 1e-10)
        ):
            raise RuntimeError(
                f"Inaccurate solution for ROI pair {pair}: "
                f"relative residual={relative_residual:.3g}"
            )

        currents = structural_graph.conductance * (
            potentials[structural_graph.edge_i]
            - potentials[structural_graph.edge_j]
        )

        edge_currents += np.abs(currents)

    return edge_currents
