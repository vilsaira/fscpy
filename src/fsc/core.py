#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Function-to-Structure Coupling (FSC)

This module implements the constrained-Laplacian / Modified Nodal Analysis
(MNA) formulation described in the accompanying paper.

Notation follows the paper:
- FC values define pairwise imposed potential-difference constraints
- SC is the structural weight matrix (weighted adjacency)
- L is the graph Laplacian built directly from SC
- phi are nodal potentials
- I are edge-level flows/currents, defined as:
      I_ij = SC_ij * (phi_i - phi_j)
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import minres


class FSC:
    """
    Function-to-Structure Coupling via constrained Laplacians / MNA.

    Parameters
    ----------
    FC : np.ndarray
        Symmetric functional connectivity matrix. Nonzero upper-triangular
        entries are treated as imposed pairwise potential differences.
    SC : np.ndarray
        Symmetric structural connectivity matrix interpreted as edge weights
        (conductances).

    Notes
    -----
    The model solves the block system

        [L  B] [phi]   [  0  ]
        [C  0] [ i_s] = [s_fc ]

    where:
    - L is the graph Laplacian derived from SC
    - B is the incidence matrix of imposed FC constraints
    - C = B.T in the absence of dependent sources
    - phi are nodal potentials
    - i_s are auxiliary source currents enforcing the constraints
    - s_fc imposed pairwise potential differences derived from FC
    """

    def __init__(self, FC: np.ndarray | None = None, SC: np.ndarray | None = None):
        if FC is None:
            raise ValueError("Give functional connectivity matrix FC.")
        if SC is None:
            raise ValueError("Give structural connectivity matrix SC.")

        self._validate_dimension(FC, SC, "FC", "SC")
        self._validate_square(FC, "FC")
        self._validate_square(SC, "SC")
        self._validate_symmetry(FC, "FC")
        self._validate_symmetry(SC, "SC")

        self._FC = np.array(FC, dtype=float, copy=True)
        self._SC = np.array(SC, dtype=float, copy=True)

        self._n_nodes: int = self._SC.shape[0]
        self._constraint_indices: np.ndarray | None = None
        self._n_constraints: int | None = None

        self._L: np.ndarray | None = None
        self._B: np.ndarray | None = None
        self._C: np.ndarray | None = None

        self._phi: np.ndarray | None = None
        self._source_currents: np.ndarray | None = None
        self._voltage_differences: np.ndarray | None = None
        self._edge_currents: np.ndarray | None = None

        self._solve()

    @staticmethod
    def _validate_square(matrix: np.ndarray, matrix_name: str) -> None:
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"{matrix_name} must be a square 2D matrix.")

    @staticmethod
    def _validate_symmetry(matrix: np.ndarray, matrix_name: str, atol: float = 1e-8) -> None:
        if not np.allclose(matrix, matrix.T, atol=atol):
            raise ValueError(f"{matrix_name} must be symmetric.")

    @staticmethod
    def _validate_dimension(A: np.ndarray, B: np.ndarray, A_name: str, B_name: str) -> None:
        if A.shape != B.shape:
            raise ValueError(f"{A_name} dimensions do not match {B_name}.")

    def _get_constraint_indices(self) -> np.ndarray:
        """
        Return upper-triangular indices of nonzero FC constraints.

        Only the upper triangle is used to avoid duplicating symmetric constraints,
        but both positive and negative FC values are included.
        """
        fc_upper = np.triu(self._FC, k=1)
        return np.argwhere(fc_upper != 0)

    def _build_laplacian(self, sc_matrix: np.ndarray | None = None) -> np.ndarray:
        """
        Build graph Laplacian L = D - W directly from structural weights SC.
        """
        if sc_matrix is None:
            sc_matrix = self._SC

        degrees = np.sum(sc_matrix, axis=1)
        laplacian = np.diag(degrees) - sc_matrix
        self._L = laplacian
        return laplacian

    def _build_constraint_incidence_matrix(self, constraint_indices: np.ndarray) -> np.ndarray:
        """
        Build incidence matrix B for imposed FC constraints.

        For each constrained pair (i, j), the column has:
        - +1 at i and -1 at j if FC_ij > 0
        - -1 at i and +1 at j if FC_ij < 0
        """
        n = self._n_nodes
        m = len(constraint_indices)
        B = np.zeros((n, m), dtype=float)

        for col, (i, j) in enumerate(constraint_indices):
            fc_value = self._FC[i, j]
            sign = np.sign(fc_value)
            if sign == 0:
                continue
            B[i, col] = sign
            B[j, col] = -sign

        self._B = B
        self._C = B.T
        return B

    def _solve_mna_system(self, A: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Solve the grounded MNA system.

        The first node is grounded (phi_0 = 0) by removing the first row and
        column of the full block system.
        """
        A_grounded = A[1:, 1:]
        z_grounded = z[1:]

        solution, info = minres(A_grounded, z_grounded)
        if info != 0:
            raise RuntimeError(f"MINRES did not converge successfully (info={info}).")

        return solution

    def _solve(self) -> None:
        """
        Assemble and solve the constrained-Laplacian / MNA system.
        """
        constraint_indices = self._get_constraint_indices()
        self._constraint_indices = constraint_indices
        self._n_constraints = len(constraint_indices)

        L = self._build_laplacian()
        B = self._build_constraint_incidence_matrix(constraint_indices)
        C = self._C
        D = np.zeros((self._n_constraints, self._n_constraints), dtype=float)

        A = np.block([[L, B], [C, D]])

        s_fc = np.array([self._FC[i, j] for i, j in constraint_indices], dtype=float)
        z = np.zeros(self._n_nodes + self._n_constraints, dtype=float)
        z[self._n_nodes :] = s_fc

        solution = self._solve_mna_system(A, z)

        phi = np.zeros(self._n_nodes, dtype=float)
        phi[1:] = solution[: self._n_nodes - 1]

        self._phi = phi
        self._source_currents = solution[self._n_nodes - 1 :]

        self._voltage_differences = phi[:, None] - phi[None, :]
        self._edge_currents = self._SC * self._voltage_differences
        np.fill_diagonal(self._edge_currents, 0.0)

    def get_nodal_potentials(self) -> np.ndarray:
        """
        Return nodal potentials phi.
        """
        if self._phi is None:
            raise RuntimeError("Model has not been solved yet.")        
        return self._phi.copy()

    def get_voltage_difference_matrix(self) -> np.ndarray:
        """
        Return signed nodal voltage-difference matrix:
            phi_i - phi_j
        """
        if self._voltage_differences is None:
            raise RuntimeError("Model has not been solved yet.")
        return self._voltage_differences.copy()

    def get_edge_currents(self) -> np.ndarray:
        """
        Return edge-level current / flow matrix:
            I_ij = SC_ij * (phi_i - phi_j)
        """
        if self._edge_currents is None:
            raise RuntimeError("Model has not been solved yet.")
        return self._edge_currents.copy()

    def get_graph_laplacian(self) -> np.ndarray:
        """
        Return graph Laplacian L = D - SC.
        """
        if self._L is None:
            raise RuntimeError("Model has not been solved yet.")
        return self._L.copy()

    def get_constraint_incidence_matrix(self) -> np.ndarray:
        """
        Return incidence matrix B for FC constraints.
        """
        if self._B is None:
            raise RuntimeError("Model has not been solved yet.")
        return self._B.copy()

    def get_source_currents(self) -> np.ndarray:
        """
        Return auxiliary MNA source currents i_s.
        """
        if self._source_currents is None:
            raise RuntimeError("Model has not been solved yet.")
        return self._source_currents.copy()

    def get_streamline_currents(
        self,
        streamline_assignments: np.ndarray | None = None,
        streamline_weights: np.ndarray | None = None,
        nodal_potentials: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Compute streamline-wise currents from node assignments.

        Parameters
        ----------
        streamline_assignments : np.ndarray
            Array of shape (n_streamlines, 2), where each row contains the
            1-based node indices assigned by MRtrix.
        streamline_weights : np.ndarray
            Structural weights for each streamline. If streamlines inherit the
            parent edge weight uniformly, pass that here.
        nodal_potentials : np.ndarray, optional
            Custom nodal potentials to use instead of the solved phi.

        Returns
        -------
        np.ndarray
            Signed streamline-wise currents.
        """
        if streamline_assignments is None:
            raise ValueError("Give streamline_assignments.")
        if streamline_weights is None:
            raise ValueError("Give streamline_weights.")

        if nodal_potentials is None:
            if self._phi is None:
                raise RuntimeError("Model has not been solved yet.")
            phi = self._phi
        else:
            phi = np.asarray(nodal_potentials, dtype=float)

        if len(streamline_assignments) != len(streamline_weights):
            raise ValueError("streamline_assignments and streamline_weights must have the same length.")

        streamline_currents = np.zeros(len(streamline_weights), dtype=float)

        for idx, assignment in enumerate(streamline_assignments):
            if np.any(assignment == 0):
                continue

            inode = int(assignment[0]) - 1
            jnode = int(assignment[1]) - 1
            weight = float(streamline_weights[idx])

            if weight <= 0:
                continue

            streamline_currents[idx] = weight * (phi[inode] - phi[jnode])

        return streamline_currents


if __name__ == "__main__":
    print("No main functions implemented. Use as a class.")