cell_centroids code path
========================

This page summarizes the code path executed when calling the API
``mesh.geometry.cell_centroids`` to compute mesh cell centroids.

Overview
--------

Cell centroids in a volume mesh are computed by normalizing the divergence of
face moments (from face area vectors and face centroids) by cell volume.

- **Entry point**: ``mesh.geometry.cell_centroids()`` (user API)
- **Implementation**: ``graphlow.geometry.volume.cell_centroids(mesh)``
- **Dependencies**: ``cell_volumes``, ``face_centroids``, ``face_area_vectors``, ``map_face_to_cell(..., "div", "segment")``

Call flow
---------

.. mermaid::

   flowchart TB
       subgraph User["User API"]
           A["cell_centroids"]
       end

       subgraph Core["src/graphlow/core/geometry.py"]
           B["MeshGeometry.cell_centroids"]
           B --> C["_cell_centroids"]
       end

       subgraph Volume["src/graphlow/geometry/volume.py"]
           D["cell_centroids"]
           V["cell_volumes"]
           D --> V
           D --> G
           D --> S
           M["moment_f → map_face_to_cell → moment_c / 4Vc"]
           D --> M
       end

       subgraph GeometryWrapper["mesh.geometry"]
           G["face_centroids"]
           S["face_area_vectors"]
       end

       subgraph Surface["src/graphlow/geometry/surface.py"]
           FC["face_centroids"]
           FAV["face_area_vectors"]
           FC --> FC3D
           FAV --> FAV3D
           FC3D["_compute_3d_face"]
           FAV3D["_compute_3d_face"]
           FC3D --> FC_DISP
           FAV3D --> FAV_DISP
           FC_DISP["analytic: triangle_centroids, quad_centroids, ..."]
           FAV_DISP["analytic: triangle_area_vectors, quad_area_vectors, ..."]
       end

       subgraph Topology["src/graphlow/core/topology.py"]
           MFTC["map_face_to_cell"]
       end

       subgraph Mapping["src/graphlow/graph/mapping.py"]
           MFTC_IMPL["map_face_to_cell"]
           SEG_DIV["_segment_div_map_face_to_cell"]
           MFTC_IMPL --> SEG_DIV
       end

       subgraph Analytic["src/graphlow/geometry/methods/analytic.py"]
           AN["tetra_volume, triangle_centroids, triangle_area_vectors, ..."]
       end

       subgraph CellVol["inside cell_volumes"]
           CV_CHECK["mesh_dim"]
           CV_POLY["_compute_volume_using_divergence_theorem"]
           CV_FIXED["_VOLUME_FN"]
           V --> CV_CHECK
           CV_CHECK --> CV_POLY
           CV_CHECK --> CV_FIXED
           CV_FIXED --> AN
       end

       A --> B
       C --> D
       G --> FC
       S --> FAV
       FC_DISP --> AN
       FAV_DISP --> AN
       M --> MFTC
       MFTC --> MFTC_IMPL

Simplified sequence diagram
---------------------------

.. mermaid::

   sequenceDiagram
       participant U as User
       participant Core as core/geometry.py
       participant Vol as geometry/volume.py
       participant Surf as geometry/surface.py
       participant Analytic as geometry/methods/analytic.py
       participant Topo as core/topology.py
       participant Map as graph/mapping.py

       U->>Core: cell_centroids
       Core->>Vol: _cell_centroids

       Vol->>Vol: cell_volumes
       Vol->>Analytic: tetra_volume, hexahedron_volume, ...
       Vol-->>Vol: Vc

       Vol->>Core: face_centroids
       Core->>Surf: face_centroids
       Surf->>Analytic: triangle_centroids, quad_centroids, ...
       Surf-->>Vol: gf

       Vol->>Core: face_area_vectors
       Core->>Surf: face_area_vectors
       Surf->>Analytic: triangle_area_vectors, quad_area_vectors, ...
       Surf-->>Vol: Sf

       Vol->>Topo: map_face_to_cell
       Topo->>Map: map_face_to_cell
       Map->>Map: _segment_div_map_face_to_cell
       Map-->>Vol: moment_c

       Vol-->>Core: cell centroids
       Core-->>U: cell centroids

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
     - ``mesh.geometry.cell_centroids()``
   * - Geometry wrapper
     - ``src/graphlow/core/geometry.py``
     - ``MeshGeometry``, ``_cell_centroids``
   * - Cell centroid implementation
     - ``src/graphlow/geometry/volume.py``
     - ``cell_centroids``, ``cell_volumes``
   * - Face centroids / face area vectors
     - ``src/graphlow/geometry/surface.py``
     - ``face_centroids``, ``face_area_vectors``, ``_compute_3d_face``, ``_dispatch_*``
   * - Topology mapping
     - ``src/graphlow/core/topology.py``
     - ``MeshTopology.map_face_to_cell``
   * - Face-to-cell aggregation
     - ``src/graphlow/graph/mapping.py``
     - ``map_face_to_cell``, ``_segment_div_map_face_to_cell``
   * - Analytic formulas
     - ``src/graphlow/geometry/methods/analytic.py``
     - ``tetra_volume``, ``triangle_centroids``, ``triangle_area_vectors``, etc.

Derivation-to-implementation mapping
------------------------------------

0. Triangles and quadrilaterals (fixed cells)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Triangle area vector**

  .. math::

     \mathbf{S}_f = \frac{1}{2}(\mathbf{e}_1\times\mathbf{e}_2), \quad
     \mathbf{e}_1 = p_1-p_0, \quad \mathbf{e}_2 = p_2-p_0

  -> ``analytic.py::triangle_area_vectors``: ``0.5 * cross(v1 - v0, v2 - v0)``.

- **Triangle centroid**

  .. math::

     \mathbf{g}_f = \frac{p_0+p_1+p_2}{3}

  -> ``analytic.py::triangle_centroids``: ``(v0 + v1 + v2) / 3``.

- **Quadrilateral area vector**
  Using diagonal vectors:

  .. math::

     \mathbf{d}_1 = p_2-p_0, \quad \mathbf{d}_2 = p_3-p_1, \quad
     \mathbf{S}_f = \frac{1}{2}(\mathbf{d}_1\times\mathbf{d}_2)

  -> ``analytic.py::quad_area_vectors``: ``0.5 * cross(d1, d2)``.

- **Quadrilateral centroid**

  .. math::

     \mathbf{g}_f = \frac{p_0+p_1+p_2+p_3}{4}

  -> ``analytic.py::quad_centroids``: ``(v0 + v1 + v2 + v3) / 4``.

1. Area vector (polygon)
~~~~~~~~~~~~~~~~~~~~~~~~

For polygon faces with cyclic vertex indexing, the signed sum is:

.. math::

   \mathbf{S}_f = \frac{1}{2}\sum_{i=0}^{n-1} p_i \times p_{i+1}.

This is implemented by ``polygon_area_vectors`` in
``src/graphlow/geometry/methods/analytic.py``:

- Cyclic index reference: ``next_idx``, ``next_conn``
- Per-edge contribution: ``tri_pieces = 0.5 * cross(p_curr, p_next)``
- Per-face accumulation: ``index_add_(..., segment_ids, tri_pieces)``

2. Face centroid (polygon)
~~~~~~~~~~~~~~~~~~~~~~~~~~

``polygon_centroids`` computes centroids by splitting each face into fan
triangles around a reference point.

For each triangle:

.. math::

   \mathbf{S}_{T_i} = \frac{1}{2}\big((p_i-p_0)\times(p_{i+1}-p_0)\big), \quad
   A_{T_i} = \|\mathbf{S}_{T_i}\|,

.. math::

   \mathbf{g}_{T_i} = \frac{p_0+p_i+p_{i+1}}{3}, \quad
   A_{T_i}\mathbf{g}_{T_i} = A_{T_i}\frac{p_0+p_i+p_{i+1}}{3}.

Therefore, for the whole face:

.. math::

   A_f = \sum_i A_{T_i}, \quad
   \mathbf{g}_f = \frac{1}{A_f}\sum_i A_{T_i}\mathbf{g}_{T_i}.

Implementation mapping (``analytic.py::polygon_centroids``):

- Reference point: ``p0``
- Neighbor vertices: ``p1``, ``p2``
- Triangle area vector: ``area_vec_c_Ti = 0.5 * cross(p1 - p0, p2 - p0)``
- Triangle area: ``area_c_Ti = vector_norm(area_vec_c_Ti, ...)``
- Moment: ``moment_c_Ti = area_c_Ti * (p1 + p2 + p0) / 3``
- Face area: ``area_c.index_add_(..., area_c_Ti)``
- Face moment sum: ``moment_c.index_add_(..., moment_c_Ti)``
- Face centroid: ``centroid_c = moment_c / area_c``

3. Cell volume
~~~~~~~~~~~~~~

``cell_volumes`` in ``src/graphlow/geometry/volume.py`` computes cell volume.

- Fixed cells (tetra/hex, etc.):
  ``tetra_volume``, ``hexahedron_volume``, etc. in
  ``src/graphlow/geometry/methods/analytic.py`` (via ``_VOLUME_FN``)
- polyhedron:
  ``_compute_volume_using_divergence_theorem``

4. Cell centroid
~~~~~~~~~~~~~~~~

``src/graphlow/geometry/volume.py::cell_centroids`` implements:

.. math::

   \begin{aligned}
   \mathbf{g}_c
   &= \frac{1}{V_c} \int_{\text{cell}} \mathbf{r}\, dV \\
   &= \frac{1}{4 V_c} \int_{\text{cell}} \nabla \cdot (\mathbf{r} \otimes \mathbf{r}) \, dV
   \quad \because \nabla \cdot (\mathbf{r} \otimes \mathbf{r}) = 4 \mathbf{r} \\
   &= \frac{1}{4 V_c} \int_{S} (\mathbf{r} \otimes \mathbf{r}) \cdot \mathbf{n} \, dA \\
   &= \frac{1}{4 V_c} \int_{S} (\mathbf{r} \cdot \mathbf{n}) \, \mathbf{r} \, dA \\
   &= \frac{1}{4 V_c} \sum_f \int_{f} (\mathbf{r} \cdot \mathbf{n}_f)\, \mathbf{r} \, dA \\
   &= \frac{1}{4 V_c} \sum_f (\mathbf{g}_f \cdot \mathbf{n}_f) \int_{f} \mathbf{r} \, dA \\
   &\quad \because\ f \text{ is planar, and }
   \mathbf{r} \cdot \mathbf{n}_f = \mathbf{g}_f \cdot \mathbf{n}_f \text{ is constant on the face} \\
   &= \frac{1}{4 V_c} \sum_f (\mathbf{g}_f \cdot \mathbf{n}_f)\, A_f \mathbf{g}_f \\
   &= \frac{1}{4 V_c} \sum_f (\mathbf{g}_f \cdot \mathbf{S}_f)\, \mathbf{g}_f
   \end{aligned}

Implementation mapping:

- Face area vectors: ``mesh.geometry.face_area_vectors()``
- Face centroids: ``mesh.geometry.face_centroids()``
- Face moments: ``torch.sum(Sf * gf, dim=-1, keepdim=True) * gf``
- Cell moments: ``mesh.topology.map_face_to_cell(moment_f, "div", "segment")``
- Cell centroids: ``moment_c / (4.0 * Vc)``
