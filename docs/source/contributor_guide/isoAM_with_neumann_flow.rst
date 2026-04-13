isoAM_with_neumann code path
============================

This page summarizes the code path executed when calling
``mesh.geometry.isoAM_with_neumann`` to compute IsoAM with Neumann handling.

Overview
--------

It computes IsoAM (Isotropic Anisotropic Metric) with Neumann boundary effects.
The routine uses a point adjacency graph, weights, surface normals, and a
moment matrix to assemble raw AM and then the gradient operator.

- **Entry point**: ``mesh.geometry.isoAM_with_neumann(...)`` (user API)
- **Implementation**: ``graphlow.geometry.operator.isoAM_with_neumann(mesh, ...)``
- **Key dependencies**: ``topology.point_adjacency()``, ``_compute_normals_on_surface_points`` (internally uses ``extract_surface``, ``face_normals``, ``map_cell_to_point``), ``_compute_moment_matrix``, ``_compute_rawAM_and_moment_inv``, ``_create_grad_operator_from``

Call flow
-------------------

.. mermaid::

   flowchart TB
       subgraph User["User API"]
           A["isoAM_with_neumann"]
       end

       subgraph Core["src/graphlow/core/geometry.py"]
           B["MeshGeometry.isoAM_with_neumann"]
           B --> C["_isoAM_with_neumann"]
       end

       subgraph Operator["src/graphlow/geometry/operator.py"]
           D["isoAM_with_neumann"]
           ADJ["point_adjacency"]
           W["weights"]
           NORM["_compute_normals_on_surface_points"]
           MOM["_compute_moment_matrix"]
           RAW["_compute_rawAM_and_moment_inv"]
           GRAD["_create_grad_operator_from"]
           D --> ADJ
           D --> W
           D --> NORM
           D --> MOM
           D --> RAW
           D --> GRAD
       end

       subgraph Topology["src/graphlow/core/topology.py"]
           ADJ_IMPL["point_adjacency"]
           GET_SKEL["get_skeleton"]
           BCACHE["bcache.sparse"]
           ADJ_IMPL --> GET_SKEL
           GET_SKEL --> BCACHE
           MAP_CP["map_cell_to_point"]
       end

       subgraph SkeletonBuilder["src/graphlow/graph/skeleton_builder.py"]
           BUILD_PP["_build_pp"]
       end

       subgraph NormalsHelper["inside operator.py"]
           EXTRACT["extract_surface"]
           FACE_N["face_normals"]
           MAP_N["map_cell_to_point"]
           NORM_SCATTER["normalize, scatter_add_to_parent_point_data"]
           EXTRACT --> FACE_N
           FACE_N --> MAP_N
           MAP_N --> NORM_SCATTER
       end

       subgraph ExtractSurface["src/graphlow/core/mesh.py"]
           PV_EXT["TensorMesh.extract_surface"]
       end

       subgraph SurfaceGeo["src/graphlow/geometry/surface.py"]
           FN["face_normals"]
           FN --> SURF_FN["face_normals"]
       end

       subgraph Mapping["src/graphlow/graph/mapping.py"]
           MAP_CP_IMPL["map_cell_to_point"]
       end

       subgraph Moment["inside operator.py"]
           DIFF["_compute_moment_matrix"]
           OUTER["u⊗u index_add"]
           DIFF --> OUTER
       end

       subgraph RawAM["inside operator.py"]
           SOLVE["_compute_rawAM_and_moment_inv"]
           PACK["Cholesky solve, pack sparse"]
           SOLVE --> PACK
       end

       subgraph GradOp["inside operator.py"]
           ROWSUM["_create_grad_operator_from"]
           DIAG["row-sum-zero"]
           ROWSUM --> DIAG
       end

       A --> B
       C --> D
       ADJ --> ADJ_IMPL
       GET_SKEL --> BUILD_PP
       W --> MAP_CP
       MAP_CP --> MAP_CP_IMPL
       NORM --> EXTRACT
       EXTRACT --> PV_EXT
       FACE_N --> FN
       MAP_N --> MAP_CP_IMPL
       D --> RAW
       MOM --> RAW
       RAW --> GRAD

Simplified sequence diagram
---------------------------

This diagram shows the path when **consider_volume=True**.

.. mermaid::

   sequenceDiagram
       participant U as User
       participant Core as core/geometry.py
       participant Op as geometry/operator.py
       participant Topo as core/topology.py
       participant Skel as graph/skeleton_builder.py
       participant Mesh as core/mesh.py
       participant Surf as geometry/surface.py
       participant Map as graph/mapping.py

       U->>Core: isoAM_with_neumann
       Core->>Op: _isoAM_with_neumann

       Op->>Topo: point_adjacency
       Topo->>Skel: get_skeleton
       Skel->>Skel: _build_pp
       Topo->>Op: adj

       Op->>Core: cell_volumes
       Core-->>Op: volumes
       Op->>Topo: map_cell_to_point
       Topo->>Map: map_cell_to_point
       Map-->>Op: weights

       Op->>Op: _compute_normals_on_surface_points
       Op->>Mesh: extract_surface
       Mesh-->>Op: surf

       Op->>Surf: face_normals
       Surf-->>Op: normals_on_faces
       Op->>Topo: map_cell_to_point
       Topo->>Map: map_cell_to_point
       Map-->>Op: normals_on_points
       Op->>Mesh: scatter_add_to_parent_point_data
       Mesh-->>Op: normals_on_surface_points

       Op->>Op: _compute_moment_matrix
       Op-->>Op: moment_matrix

       Op->>Op: _compute_rawAM_and_moment_inv
       Op-->>Op: rawAM, moment_inv

       Op->>Op: _create_grad_operator_from
       Op-->>Op: NisoAM

       Op-->>Core: NisoAM, weighted_normals, moment_inv
       Core-->>U: isoam, weighted_normals, moment_inv

File and symbol mapping
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 12 32 40

   * - Role
     - File
     - Main symbols
   * - User API
     - (usage code)
     - ``mesh.geometry.isoAM_with_neumann(...)``
   * - Geometry wrapper
     - ``src/graphlow/core/geometry.py``
     - ``MeshGeometry.isoAM_with_neumann``, ``_isoAM_with_neumann``
   * - IsoAM implementation
     - ``src/graphlow/geometry/operator.py``
     - ``isoAM_with_neumann``, ``_compute_normals_on_surface_points``, ``_compute_moment_matrix``, ``_compute_rawAM_and_moment_inv``, ``_create_grad_operator_from``
   * - Topology
     - ``src/graphlow/core/topology.py``
     - ``point_adjacency``, ``get_skeleton``, ``map_cell_to_point``
   * - Skeleton construction
     - ``src/graphlow/graph/skeleton_builder.py``
     - ``_build_pp`` (PP adjacency)
   * - Surface extraction
     - ``src/graphlow/core/mesh.py``
     - ``TensorMesh.extract_surface``
   * - Face normals
     - ``src/graphlow/geometry/surface.py``
     - ``face_normals`` (called on ``surf``)
   * - Cell-to-point mapping
     - ``src/graphlow/graph/mapping.py``
     - ``map_cell_to_point``

Notes
-----

- **PP**: Point-to-point adjacency (vertex pairs connected by an edge). It is cached via ``get_skeleton(PP)`` and converted to backend sparse COO by ``bcache.sparse(..., "coo")``.
- **consider_volume=True**: Only in this mode, ``cell_volumes()`` and ``map_cell_to_point(volumes, ...)`` are used, and their result becomes vertex weights.
- **normals_on_surface_points**: Normals are nonzero only on the *surface* of the volume mesh. The flow is: extract a surface mesh with ``extract_surface()``, map its ``face_normals`` to points via ``map_cell_to_point``, then scatter the result back to the parent volume point space via ``surf.scatter_add_to_parent_point_data(...)`` and normalize.
- **moment_matrix**: Weighted sum of :math:`u_{ij} \otimes u_{ij}` over adjacency vectors :math:`u_{ij}`. In the Neumann variant, ``n_otimes_n`` (outer product of surface normals) is added to reduce singularity risk.
- **rawAM / NisoAM**: rawAM is a scalar-weighted form of :math:`M_i^{-1}(x_j-x_i)` represented as sparse COO with shape ``(K, N, N)``. ``_create_grad_operator_from`` converts it into a row-sum-zero gradient operator.
