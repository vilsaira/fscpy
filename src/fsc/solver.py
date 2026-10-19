"""Function-structure coupling through a constrained graph Laplacian."""

from dataclasses import dataclass
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, minres, spsolve


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


def solve_fsc(structural_graph, functional_connectivity, *, ground_node=0,
              solver="auto", rtol=1e-8, maxiter=None):
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
    if not 0 <= ground_node < n_nodes:
        raise ValueError("Invalid ground node.")
    if solver not in ("auto", "direct", "minres"):
        raise ValueError(f"Unknown solver: {solver}")

    if n_constraints == 0:
        return FSCResult(
            np.zeros(n_nodes), np.zeros(len(structural_graph.edge_i)),
            np.empty(0), np.empty(0), np.zeros(n_nodes), 0.0,
        )

    laplacian = structural_graph.laplacian()
    matrix = sparse.bmat([[laplacian, incidence], [incidence.T, None]]).tocsr()
    rhs = np.concatenate((np.zeros(n_nodes), constraints))

    mask = np.ones(n_nodes + n_constraints, dtype=bool)
    mask[ground_node] = False
    reduced_matrix = matrix[mask][:, mask]
    reduced_rhs = rhs[mask]
    solution = np.zeros(n_nodes + n_constraints)

    if solver == "auto":
        solver = "direct" if len(reduced_rhs) <= 1000 else "minres"

    if solver == "direct":
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                solution[mask] = spsolve(reduced_matrix.tocsc(), reduced_rhs)
        except MatrixRankWarning as exc:
            raise ValueError("Singular MNA system: check functional constraints.") from exc
    else:
        solution[mask], status = minres(
            reduced_matrix, reduced_rhs, rtol=rtol, maxiter=maxiter
        )
        if status != 0:
            raise RuntimeError(f"MINRES did not converge (status={status}).")

    potentials = solution[:n_nodes]
    source_currents = solution[n_nodes:]
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
            f"{solver} returned an inaccurate FSC solution "
            f"(relative residual={relative_residual:.3g})."
        )

    return FSCResult(
        potentials, edge_currents, source_currents,
        constraint_residual, current_balance, relative_residual,
    )
