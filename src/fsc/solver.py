"""Function-structure coupling using modified nodal analysis."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import minres


@dataclass
class FSCResult:
    potentials: np.ndarray
    edge_currents: np.ndarray
    source_currents: np.ndarray
    constraint_residual: np.ndarray
    current_balance: np.ndarray
    relative_residual: float


def constraints_from_fc(fc, node_ids=None):
    """Return nonzero FC pairs and values from the upper triangle of an FC matrix.

    When node_ids is omitted, pairs contain zero-based indices into the FC
    matrix. Otherwise, those indices are replaced by graph node IDs.
    """
    fc = np.asarray(fc, dtype=float)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError("FC must be square.")

    if not np.allclose(fc, fc.T, equal_nan=True):
        raise ValueError("FC must be symmetric.")

    selected = np.triu(np.isfinite(fc) & (fc != 0), k=1)
    pairs = np.column_stack(np.nonzero(selected))
    values = fc[pairs[:, 0], pairs[:, 1]]

    if node_ids is not None:
        pairs = np.asarray(node_ids, dtype=int)[pairs]

    return pairs, values


def solve_fsc(graph, constraints, fc_constraints, *, ground_node=0,
              rtol=1e-8, maxiter=None):
    """Solve FSC on an undirected weighted graph.

    Parameters
    ----------
    graph : WeightedGraph
        Structural graph.
    constraints : ndarray, shape (m, 2), or sparse matrix, shape (n_nodes, m)
        Either pairs of graph node indices, or a constraint matrix whose
        columns contain coefficients for linear combinations of node
        potentials. Each column defines one functional constraint.
    fc_constraints : array_like, shape (m,)
        FC values. Their absolute magnitudes are imposed as potential
        differences or weighted parcel-potential differences.
    ground_node : int, default=0
        Graph node whose potential is fixed at zero.
    rtol : float, default=1e-8
        Relative convergence tolerance for MINRES.
    maxiter : int, optional
        Maximum number of MINRES iterations.

    Returns
    -------
    FSCResult
        Potentials, edge and source currents, and residual diagnostics.
    """
    n_nodes = graph.n_nodes
    values = np.abs(np.asarray(fc_constraints, dtype=float).reshape(-1))
    n_constraints = len(values)
    if not np.all(np.isfinite(values)):
        raise ValueError("Functional constraint values must be finite.")

    if sparse.issparse(constraints):
        incidence = constraints.tocsr()
        if incidence.shape != (n_nodes, n_constraints):
            raise ValueError("Constraint matrix has incompatible dimensions.")
    else:
        pairs = np.asarray(constraints, dtype=int).reshape(-1, 2)
        if len(pairs) != n_constraints:
            raise ValueError("Constraint pairs and values must have matching lengths.")
        if np.any(pairs < 0) or np.any(pairs >= n_nodes):
            raise ValueError("Constraint node outside graph.")

        rows = np.empty(2 * n_constraints, dtype=int)
        columns = np.empty(2 * n_constraints, dtype=int)
        coefficients = np.empty(2 * n_constraints, dtype=float)

        for constraint_id, (source, target) in enumerate(pairs):
            rows[2 * constraint_id:2 * constraint_id + 2] = (source, target)
            columns[2 * constraint_id:2 * constraint_id + 2] = constraint_id
            coefficients[2 * constraint_id:2 * constraint_id + 2] = (1.0, -1.0)

        incidence = sparse.coo_matrix(
            (coefficients, (rows, columns)), shape=(n_nodes, n_constraints)
        ).tocsr()

    if not 0 <= ground_node < n_nodes:
        raise ValueError("Invalid ground node.")

    if n_constraints == 0:
        return FSCResult(
            np.zeros(n_nodes), np.zeros(len(graph.edge_i)),
            np.empty(0), np.empty(0), np.zeros(n_nodes), 0.0
        )

    laplacian = graph.laplacian()
    matrix = sparse.bmat([[laplacian, incidence], [incidence.T, None]]).tocsr()
    rhs = np.concatenate((np.zeros(n_nodes), values))

    mask = np.ones(n_nodes + n_constraints, dtype=bool)
    mask[ground_node] = False
    solution = np.zeros(n_nodes + n_constraints)
    solution[mask], status = minres(
        matrix[mask][:, mask], rhs[mask], rtol=rtol, maxiter=maxiter
    )
    if status != 0:
        raise RuntimeError(f"MINRES did not converge (status={status}).")

    potentials = solution[:n_nodes]
    source_currents = solution[n_nodes:]
    edge_currents = graph.conductance * (
        potentials[graph.edge_i] - potentials[graph.edge_j]
    )
    constraint_residual = incidence.T @ potentials - values
    current_balance = laplacian @ potentials + incidence @ source_currents

    residual = np.linalg.norm(np.concatenate((current_balance, constraint_residual)))
    scale = np.linalg.norm(values)
    relative_residual = float(residual / scale) if scale else float(residual)

    return FSCResult(
        potentials, edge_currents, source_currents,
        constraint_residual, current_balance, relative_residual
    )
