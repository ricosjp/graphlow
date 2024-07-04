
import numpy as np
import pyvista as pv
from scipy import sparse as sp

from graphlow.util import cache


class GraphlowMesh:

    def __init__(self, mesh: pv.PointGrid):
        self.mesh = mesh
        return

    @cache.cache
    def compute_cell_point_incidence(self) -> sp.csr_array:
        """Compute (n_cells, n_points)-shaped sparse incidence matrix.
        The method is cached.

        Returns
        -------
        scipy.sparse.csr_array[bool]
            (n_cells, n_points)-shapece sparse incidence matrix.
        """
        indices = self.mesh.cell_connectivity
        indptr = self.mesh.offset
        data = np.ones(len(indices), dtype=bool)
        shape = (self.mesh.n_cells, self.mesh.n_points)

        return sp.csr_array((data, indices, indptr), shape=shape)

    @cache.cache
    def compute_cell_adjacency(self) -> sp.csr_array:
        """Compute (n_cells, n_cells)-shaped sparse adjacency matrix including
        self-loops. The method is cached.

        Returns
        -------
        scipy.sparse.csr_array[bool]
            (n_cells, n_cells)-shapece sparse adjacency matrix.
        """
        cp_inc = self.compute_cell_point_incidence()
        cc_adj = cp_inc @ cp_inc.T
        return cc_adj

    @cache.cache
    def compute_point_adjacency(self) -> sp.csr_array:
        """Compute (n_points, n_points)-shaped sparse adjacency matrix
        including self-loops. The method is cached.

        Returns
        -------
        scipy.sparse.csr_array[bool]
            (n_points, n_points)-shapece sparse adjacency matrix.
        """
        cp_inc = self.compute_cell_point_incidence()
        pp_adj = cp_inc.T @ cp_inc
        return pp_adj
