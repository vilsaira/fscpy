#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
The FSC class contains methods needed for the Function-to-Structure Coupler
(FSC). FSC can be used to combine functional and structural
connectivity matrices[1] obtained from different imaging modalities.

Author: Viljami Sairanen

[1] Viljami Sairanen, Combining function and structure in a single macro-scale
connectivity model of the human brain, bioarxiv


Example of usage with the Human Connectome Project from the ENIGMA TOOLBOX.

from enigmatoolbox.datasets import load_sc, load_fc
from nilearn import plotting
from function_to_structure_coupler import FSC

# Load cortico-cortical functional connectivity data
fc_ctx, fc_ctx_labels, _, _ = load_fc() 
# Load cortico-cortical structural connectivity data
sc_ctx, sc_ctx_labels, _, _ = load_sc() 
# Calculate cortico-cortical 'functio-structural current' connectivity data
fscc_ctx = FSC(V=fc_ctx, R=sc_ctx).get_nodal_currents_I()

#  # Plot cortico-cortical connectivity matrices
fc_plot = plotting.plot_matrix(fc_ctx, figure=(9, 9), labels=fc_ctx_labels, vmax=0.8, vmin=0, cmap='Reds')
sc_plot = plotting.plot_matrix(sc_ctx, figure=(9, 9), labels=sc_ctx_labels, vmax=10, vmin=0, cmap='Blues')
fsc_plot = plotting.plot_matrix(fscc_ctx, figure=(9, 9), labels=sc_ctx_labels, vmax=fscc_ctx.max(), vmin=fscc_ctx.min(), cmap='Greens')

"""

import numpy as np
import scipy
import sys

class FSC(object):

    def __init__(self, V=None, R=None):
        if V is None:
            raise ValueError("Give source voltages V")
        if R is None:
            raise ValueError("Give resistances R")
        
        self._V = V
        self._R = R
        self._U = np.zeros_like(V) # nodal voltage difference matrix
        self._calculate_U = True # flag if U needs to be calculated on get        
        self._G = None # Conductance matrix
        self._B = None # Incidence matrix
        self._C = None # often B.T
        self._n = None # number of edges
        self._m = None # number of voltage sources
        self._nodal_voltages = None
        self._source_currents = None # i_s

        self._update_nodal_voltages_U()
        # self.get_nodal_currents_I()
        # self.get_voltage_difference_matrix_U()
        # self.get_conductance_matrix_G()
        # self.get_incidence_matrix_B()
        # self.get_incidence_matrix_C()
        # self._calculate_conductance_matrix_G(resistance_matrix=None)
        # self._calculate_incidence_matrix_B()
        # self.predict_source_voltages(resistance_matrix=None)

    def _validateSymmetry(self, matrix, matrix_name):
        if (np.abs(matrix - matrix.T).max() > 1e-8):
            sys.exit(f"Error in setting {matrix_name}: {matrix_name} is not symmetric along diagonal.")
        return 0
    
    def _validateDimension(self, A, B, A_name, B_name):
        if (A is not None) & (B is not None):
            if (A.shape != B.shape):
                sys.exit(f"Error in setting {A_name}: dimensions don't match with {B_name}.")

    def get_voltage_difference_matrix_U(self):
        if self._calculate_U:
            self._update_nodal_voltages_U()
            self._calculate_U = False
        return self._U - self._U.T
    
    def get_nodal_currents_I(self):
        I = self.get_voltage_difference_matrix_U() / self._R
        np.fill_diagonal(I, 0)
        I[np.isnan(I)] = 0.0
        return I
    
    def get_conductance_matrix_G(self):
        if self._G is None:
            self._update_nodal_voltages_U()
        return self._G
    
    def get_incidence_matrix_B(self):
        if self._B is None:
            self._update_nodal_voltages_U()
        return self._B
    
    def get_incidence_matrix_C(self):
        if self._C is None:
            self._update_nodal_voltages_U()
        return self._C
    
    def get_nodal_voltages(self):
        if self._nodal_voltages is None:
            self._update_nodal_voltages_U()
        return self._nodal_voltages
    
    def get_streamline_currents(self, streamline_assignments=None, streamline_resistances=None, nodal_voltages=None):
        # streamline assignments detail the origin and end node of a given tract
        # streamline resistances detail the edge weight between the origin
        # and end node of a given tract.
        # nodal_voltages is an optional input that can replace the object's own
        # nodal voltage U
        U = self._U
        if nodal_voltages is not None:
            U = nodal_voltages

        streamline_currents = np.zeros_like(streamline_resistances)
        for i, assignment in enumerate(streamline_assignments):
            if any(assignment == 0):
                # No connection between the two nodes
                continue
            inode = assignment[0].astype(int) - 1 # Note, indexing in MRtrix streamline assignments starts from 1 so shift by one to left.
            fnode = assignment[1].astype(int) - 1 
            streamline_currents[i] = U[inode, fnode] / streamline_resistances[i]
        return streamline_currents

    def _calculate_conductance_matrix_G(self, resistance_matrix=None):
        # Calculate conductance matrix G from resistance matrix R
        if resistance_matrix is None:
            resistance_matrix = self._R
        n = resistance_matrix.shape[0]
        G = np.zeros((n,n)) # Conductance matrix G: 
        # G_ii is the sum of all conductances connected to node i
        # G_ij is the negative of the sum of all conductances connected from i to j
        for i in range(n):
            for j in range(n):
                if resistance_matrix[i,j] > 0:
                    G[i,i] += 1 / resistance_matrix[i,j]
                    G[i,j] -= 1 / resistance_matrix[i,j]
        self._G = G
        return G

    def _calculate_incidence_matrix_B(self, voltage_source_indices):
        
        n = self._n # nodes
        m = self._m # voltage sources
        B = np.zeros((n,m))
        # If the positive/negative terminal of the i-th voltage source is connected to node j, then the element (i,j) in the B matrix is 1 / -1.
        for i, inds in enumerate(voltage_source_indices):
            volt = self._V[inds[0], inds[1]]
            if volt != 0:
                B[inds[0], i] = np.sign(volt)
                B[inds[1], i] = -np.sign(volt)

        self._B = B
        return B
    
    def _calculate_nodal_voltages_and_source_currents(self, A, z):
        # Set the first node as the ground and remove corresponding elements from A and z
        A1 = A[1:,1:]
        z1 = z[1:]
        #return scipy.linalg.lstsq(A1, z1)[0] # nodal voltage and source current
        #vector
        # return linalg.solve(A1 + 1e-12 * np.eye(A1.shape[0]), z1, assume_a='sym') # nodal voltage and source current vector
        return scipy.sparse.linalg.minres(A1, z1)[0]

    def _update_nodal_voltages_U(self):
        # Modified Nodal Analysis for the given V and R matrices
        # returns nodal voltages U
        voltage_source_indices = np.argwhere(self._V > 0)
        # voltage_source_indices = np.argwhere(np.triu(np.ones_like(self._V),k=1))
        m = len(voltage_source_indices)
        n = self._R.shape[0] # number of nodes or vertices
        self._m = m
        self._n = n
        
        G = self._calculate_conductance_matrix_G()
        B = self._calculate_incidence_matrix_B(voltage_source_indices)
        C = B.T # If and only if there exist no dependent sources! In our analogy such are not present and C is the transpose of B.
        self._C = C
        D = np.zeros((m,m)) # If and only if there exist no dependent sources!

        # A = [[G, B], [C, D]]
        A = np.block([[G, B], [C, D]])

        z = np.zeros((n+m,)) # There are no current sources thus sum of entering/leaving currents in each node must be zero so the first n elements of z are zero
        for i in range(m):
            volt = self._V[voltage_source_indices[i][0], voltage_source_indices[i][1]]
            z[i+n] = volt

        u_i = self._calculate_nodal_voltages_and_source_currents(A, z)

        # Calculate voltage differences. The first node has voltage of 0 as it is ground.
        nodal_voltages = np.zeros((1,))
        nodal_voltages = np.concatenate((nodal_voltages, u_i[0:n-1]), axis=0)
        for i in range(n-1):
            for j in range(i+1, n):
                if self._R[i,j] > 0:
                    self._U[i,j] = np.abs(nodal_voltages[j] - nodal_voltages[i])

        self._nodal_voltages = nodal_voltages
        self._source_currents = u_i[n-1:] # -1 due to grounding done by removing the first element
        
        # Nodal voltages have been updated successfully
        self._calculate_U = False
        
        return 1
    
    @property
    def _V(self):
        return self.__V
    
    @_V.setter
    def _V(self, value):
        self.__V = np.triu(value)

    @property
    def _R(self):
        return self.__R
    
    @_R.setter
    def _R(self, value):
        # self._validateSymmetry(value, "R")
        self._validateDimension(value, self._V, "R", "V")
        self.__R = value

    @property
    def _U(self):
        return self.__U
    
    @_U.setter
    def _U(self, value):
        self.__U = value

    @property
    def _B(self):
        return self.__B
    
    @_B.setter
    def _B(self, value):
        self.__B = value

    @property
    def _C(self):
        return self.__C
    
    @_B.setter
    def _C(self, value):
        self.__C = value

    @property
    def _G(self):
        return self.__G
    
    @_G.setter
    def _G(self, value):
        self.__G = value

    @property
    def _m(self):
        return self.__m
    
    @_m.setter
    def _m(self, value):
        self.__m = value

    @property
    def _n(self):
        return self.__n
    
    @_n.setter
    def _n(self, value):
        self.__n = value

    @property
    def _nodal_voltages(self):
        return self.__nodal_voltages
    
    @_nodal_voltages.setter
    def _nodal_voltages(self, value):
        self.__nodal_voltages = value

if __name__ == "__main__":
    print("No main functions implemented. Use as a class.")
