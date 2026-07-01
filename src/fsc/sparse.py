#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""Sparse FSC / MNA backend.

This is a sparse counterpart of :mod:`fsc.core`.  It keeps the same idea as the
original FSC implementation: FC entries define imposed potential-difference
constraints, SC is interpreted as conductance, and edge currents are computed as
conductance times voltage difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import minres

ConstraintMode = Literal["fsc", "direct"]


@dataclass(frozen=True)
class SparseFSCResult:
    """Solved sparse FSC result."""

    phi: np.ndarray
    source_currents: np.ndarray
    edge_currents: np.ndarray
    constraint_i: np.ndarray
    constraint_j: np.ndarray
    constraint_values: np.ndarray
    laplacian: sparse.csr_matrix
    incidence: sparse.csr_matrix
    mna_matrix: sparse.csr_matrix
    ground_node: int

    def effective_conductance(self, eps: float = 1e-15) -> np.ndarray:
        """Return ``|source_current| / |constraint_value|`` per constraint."""
        denom = np.abs(self.constraint_values)
        out = np.full(denom.shape, np.nan, dtype=float)
        keep = denom > eps
        out[keep] = np.abs(self.source_currents[keep]) / denom[keep]
        return out

    def effective_resistance(self, eps: float = 1e-15) -> np.ndarray:
        """Return reciprocal effective conductance per constraint."""
        g = self.effective_conductance(eps=eps)
        out = np.full(g.shape, np.nan, dtype=float)
        keep = np.isfinite(g) & (g > eps)
        out[keep] = 1.0 / g[keep]
        return out


def build_laplacian_from_edges(n_nodes: int, edge_i, edge_j, conductance) -> sparse.csr_matrix:
    """Build sparse undirected graph Laplacian ``L = D - W`` from edge arrays."""
    edge_i = np.asarray(edge_i, dtype=np.int64)
    edge_j = np.asarray(edge_j, dtype=np.int64)
    conductance = np.asarray(conductance, dtype=float)

    if n_nodes <= 0:
        raise ValueError("n_nodes must be positive.")
    if edge_i.ndim != 1 or edge_j.ndim != 1 or conductance.ndim != 1:
        raise ValueError("edge_i, edge_j and conductance must be 1D arrays.")
    if not (edge_i.size == edge_j.size == conductance.size):
        raise ValueError("edge_i, edge_j and conductance must have equal length.")

    keep = (
        np.isfinite(conductance) & (conductance > 0) &
        (edge_i >= 0) & (edge_i < n_nodes) &
        (edge_j >= 0) & (edge_j < n_nodes) &
        (edge_i != edge_j)
    )
    ei = edge_i[keep]
    ej = edge_j[keep]
    g = conductance[keep]

    rows = np.concatenate([ei, ej, ei, ej])
    cols = np.concatenate([ei, ej, ej, ei])
    data = np.concatenate([g, g, -g, -g])
    L = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    L.sum_duplicates()
    L.eliminate_zeros()
    return L


def _valid_constraints(n_nodes: int, ci, cj, values):
    ci = np.asarray(ci, dtype=np.int64)
    cj = np.asarray(cj, dtype=np.int64)
    values = np.asarray(values, dtype=float)
    if ci.ndim != 1 or cj.ndim != 1 or values.ndim != 1:
        raise ValueError("constraint arrays must be 1D.")
    if not (ci.size == cj.size == values.size):
        raise ValueError("constraint arrays must have equal length.")
    keep = (
        np.isfinite(values) & (values != 0) &
        (ci >= 0) & (ci < n_nodes) &
        (cj >= 0) & (cj < n_nodes) &
        (ci != cj)
    )
    return ci[keep], cj[keep], values[keep]


def build_constraint_incidence(n_nodes: int, constraint_i, constraint_j, constraint_values,
                               mode: ConstraintMode = "fsc") -> sparse.csr_matrix:
    """Build sparse MNA incidence matrix B for functional constraints.

    ``mode='fsc'`` reproduces the sign convention of the original dense FSC
    class. ``mode='direct'`` imposes ``phi_i - phi_j = FC_ij`` directly.
    """
    ci, cj, values = _valid_constraints(n_nodes, constraint_i, constraint_j, constraint_values)
    if mode not in ("fsc", "direct"):
        raise ValueError("mode must be 'fsc' or 'direct'.")
    signs = np.sign(values) if mode == "fsc" else np.ones_like(values)
    m = values.size
    rows = np.concatenate([ci, cj])
    cols = np.concatenate([np.arange(m), np.arange(m)])
    data = np.concatenate([signs, -signs])
    B = sparse.coo_matrix((data, (rows, cols)), shape=(n_nodes, m)).tocsr()
    B.sum_duplicates()
    B.eliminate_zeros()
    return B


def constraints_from_fc_matrix(fc_matrix: np.ndarray, node_ids: np.ndarray | None = None):
    """Convert nonzero upper-triangular FC entries into constraint arrays."""
    fc = np.asarray(fc_matrix, dtype=float)
    if fc.ndim != 2 or fc.shape[0] != fc.shape[1]:
        raise ValueError("fc_matrix must be square.")
    if not np.allclose(fc, fc.T, atol=1e-8, equal_nan=True):
        raise ValueError("fc_matrix must be symmetric.")
    pairs = np.argwhere(np.triu(np.isfinite(fc) & (fc != 0), k=1))
    ci = pairs[:, 0].astype(np.int64)
    cj = pairs[:, 1].astype(np.int64)
    values = fc[ci, cj].astype(float)
    if node_ids is not None:
        node_ids = np.asarray(node_ids, dtype=np.int64)
        if node_ids.shape != (fc.shape[0],):
            raise ValueError("node_ids must have one entry per FC node.")
        ci = node_ids[ci]
        cj = node_ids[cj]
    return ci, cj, values


class SparseFSC:
    """Sparse constrained-Laplacian / MNA solver."""

    def __init__(self, n_nodes: int, edge_i, edge_j, conductance,
                 constraint_i, constraint_j, constraint_values,
                 constraint_mode: ConstraintMode = "fsc",
                 ground_node: int = 0, rtol: float = 1e-8,
                 maxiter: int | None = None):
        self.n_nodes = int(n_nodes)
        self.edge_i = np.asarray(edge_i, dtype=np.int64)
        self.edge_j = np.asarray(edge_j, dtype=np.int64)
        self.conductance = np.asarray(conductance, dtype=float)
        self.constraint_i = np.asarray(constraint_i, dtype=np.int64)
        self.constraint_j = np.asarray(constraint_j, dtype=np.int64)
        self.constraint_values = np.asarray(constraint_values, dtype=float)
        self.constraint_mode = constraint_mode
        self.ground_node = int(ground_node)
        self.rtol = float(rtol)
        self.maxiter = maxiter
        self._result: SparseFSCResult | None = None

    def solve(self) -> SparseFSCResult:
        if self.ground_node < 0 or self.ground_node >= self.n_nodes:
            raise ValueError("ground_node is out of bounds.")

        ci, cj, values = _valid_constraints(
            self.n_nodes, self.constraint_i, self.constraint_j, self.constraint_values
        )
        if values.size == 0:
            raise ValueError("No valid nonzero constraints were supplied.")

        L = build_laplacian_from_edges(self.n_nodes, self.edge_i, self.edge_j, self.conductance)
        B = build_constraint_incidence(self.n_nodes, ci, cj, values, mode=self.constraint_mode)
        D = sparse.csr_matrix((B.shape[1], B.shape[1]), dtype=float)
        A = sparse.bmat([[L, B], [B.T, D]], format="csr")

        z = np.zeros(self.n_nodes + B.shape[1], dtype=float)
        z[self.n_nodes:] = values

        # Ground one graph node to remove the Laplacian nullspace.
        keep = np.ones(A.shape[0], dtype=bool)
        keep[self.ground_node] = False
        Ag = A[keep][:, keep]
        zg = z[keep]

        try:
            sol_g, info = minres(Ag, zg, rtol=self.rtol, maxiter=self.maxiter)
        except TypeError:  # older SciPy
            sol_g, info = minres(Ag, zg, tol=self.rtol, maxiter=self.maxiter)
        if info != 0:
            raise RuntimeError(f"Sparse MINRES did not converge successfully (info={info}).")

        sol = np.zeros(A.shape[0], dtype=float)
        sol[keep] = sol_g
        phi = sol[:self.n_nodes]
        source_currents = sol[self.n_nodes:]
        edge_currents = self.conductance * (phi[self.edge_i] - phi[self.edge_j])

        self._result = SparseFSCResult(
            phi=phi,
            source_currents=source_currents,
            edge_currents=edge_currents,
            constraint_i=ci,
            constraint_j=cj,
            constraint_values=values,
            laplacian=L,
            incidence=B,
            mna_matrix=A,
            ground_node=self.ground_node,
        )
        return self._result

    def get_result(self) -> SparseFSCResult:
        return self.solve() if self._result is None else self._result
