.. _migration-0-1:

Migrating to 0.1.x mesh API
===========================

This page maps the **legacy** ``GraphlowMesh``-style API (``mesh.compute_*``,
``mesh.dict_*_tensor``, top-level ``mesh.device``, and similar) to the
**current** :class:`~graphlow.core.mesh.TensorMesh` layout: ``mesh.topology``,
``mesh.geometry``, ``mesh.backend``, and attribute containers.

For the high-level release summary (removed packages, new modules, tests, and
docs), see :doc:`../release_notes`.

The tables below use ``mesh`` for a :class:`~graphlow.core.mesh.TensorMesh`
instance. Names that are unchanged may still differ in type, caching, or
backend behavior—check the API reference when porting.

Mesh core (points and cells)
----------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.pvmesh``
     - ``mesh.pvmesh``
   * - ``mesh.points``
     - ``mesh.points``
   * - ``mesh.n_points``
     - ``mesh.n_points``
   * - ``mesh.n_cells``
     - ``mesh.n_cells``

Feature dictionaries
--------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.dict_point_tensor``
     - ``mesh.point_data``
   * - ``mesh.dict_cell_tensor``
     - ``mesh.cell_data``
   * - ``mesh.dict_sparse_tensor``
     - **Removed.** Topology is cached on ``mesh.topology`` (no separate
       sparse dict on the mesh).

Backend (device and dtypes)
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.float_precision``
     - ``mesh.backend.float_precision``
   * - ``mesh.device``
     - ``mesh.backend.device``
   * - ``mesh.dtype``
     - ``mesh.backend.dtype``

I/O and PyVista
---------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.save``
     - ``mesh.save``
   * - ``mesh.send``
     - **Removed.** Choose the execution device via ``mesh.backend`` / backend
       construction instead of a dedicated ``send`` helper.
   * - ``mesh.copy_features_from_pyvista``
     - ``mesh.copy_features_from_pyvista``
   * - ``mesh.copy_features_to_pyvista``
     - ``mesh.copy_features_to_pyvista``

Extraction and indexing
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.add_original_index``
     - **Removed.** Parent/surface index semantics are handled inside
       ``mesh.extract_surface``.
   * - ``mesh.extract_surface``
     - ``mesh.extract_surface``
   * - ``mesh.extract_cells``
     - **Removed.** Use ``mesh.topology.cell_block`` for per-cell-type
       connectivity and filtering workflows.
   * - ``mesh.extract_facets``
     - **Removed.** Use ``mesh.topology.face_block`` for face-type connectivity
       and related workflows.

Cell-point maps and medians
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.convert_elemental2nodal``
     - ``mesh.topology.map_cell_to_point``
   * - ``mesh.convert_nodal2elemental``
     - ``mesh.topology.map_point_to_cell``
   * - ``mesh.compute_median``
     - ``mesh.topology.median_points`` or ``mesh.topology.median_cells``
       (pick by whether you aggregate at points or cells).

Geometry (``compute_*`` → ``mesh.geometry``)
--------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.compute_area_vecs``
     - ``mesh.geometry.face_area_vectors``
   * - ``mesh.compute_areas``
     - ``mesh.geometry.face_areas``
   * - ``mesh.compute_volumes``
     - ``mesh.geometry.cell_volumes``
   * - ``mesh.compute_normals``
     - ``mesh.geometry.face_normals``
   * - ``mesh.compute_surface_volume``
     - ``mesh.geometry.surface_volume``
   * - ``mesh.compute_isoAM``
     - ``mesh.geometry.isoAM``
   * - ``mesh.compute_isoAM_with_neumann``
     - ``mesh.geometry.isoAM_with_neumann``

Topology and adjacency (``compute_*`` → ``mesh.topology``)
----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Legacy
     - New
   * - ``mesh.compute_cell_point_incidence``
     - ``mesh.topology.cell_point_incidence``
   * - ``mesh.compute_cell_adjacency``
     - ``mesh.topology.cell_adjacency``
   * - ``mesh.compute_point_adjacency``
     - ``mesh.topology.point_adjacency``
   * - ``mesh.compute_point_degree``
     - ``mesh.topology.point_degree_matrix``
   * - ``mesh.compute_cell_degree``
     - ``mesh.topology.cell_degree_matrix``
   * - ``mesh.compute_normalized_point_adjacency``
     - ``mesh.topology.normalized_point_adjacency``
   * - ``mesh.compute_normalized_cell_adjacency``
     - ``mesh.topology.normalized_cell_adjacency``
   * - ``mesh.compute_facet_cell_incidence``
     - ``mesh.topology.face_cell_incidence``

Removed helpers (rewrite pattern)
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Legacy
     - Migration strategy
   * - ``mesh.compute_point_relative_incidence``
     - Use ``extract_surface`` together with ``gather_parent_point_data`` /
       ``scatter_add_to_parent_point_data``.
   * - ``mesh.compute_cell_relative_incidence``
     - Reconstruct from the point-level correspondence plus the appropriate
       ``mesh.topology`` maps (cell-point / face-cell as needed).


Suggested porting order
-----------------------

1. Replace construction/IO with :func:`~graphlow.read` /
   :func:`~graphlow.from_pyvista` and an explicit backend.
2. Rename ``dict_*_tensor`` access to ``point_data`` / ``cell_data``.
3. Replace ``mesh.device`` / ``mesh.dtype`` reads with ``mesh.backend.*``.
4. Batch-replace ``compute_*`` geometry calls with ``mesh.geometry.*`` and
   topology calls with ``mesh.topology.*`` using the tables above.
5. For removed extraction or relative-incidence helpers, sketch the data flow
   with ``cell_block`` / ``face_block`` and parent-surface gather/scatter
   before deleting old glue code.
