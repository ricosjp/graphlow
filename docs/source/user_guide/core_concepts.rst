Core concepts
=============

``graphlow`` becomes easier to use once you separate three concerns:
the mesh container, geometry computations, and topology-derived operators.

``TensorMesh`` is the central object
------------------------------------

The main runtime object is ``TensorMesh``. It keeps:

- mesh coordinates in backend tensors
- point and cell data in backend tensors
- the original PyVista mesh as the topology source
- accessors for geometry and topology functionality

This design lets the library preserve the mesh structure from PyVista while
running differentiable tensor computations on coordinates and data arrays.

Geometry vs topology
--------------------

``mesh.geometry`` handles coordinate-dependent quantities such as:

- face normals
- centroids
- areas
- volumes

These values depend on point coordinates and therefore participate in autograd
workflows.

``mesh.topology`` handles connectivity-derived structures such as:

- incidence matrices
- adjacency matrices
- Laplacians
- mapping and aggregation operators

These structures depend on mesh connectivity rather than point positions and
are therefore good candidates for caching.

If you want concrete tutorials for these ideas, the example gallery now starts
with beginner examples for ``TensorMesh`` creation, cell volumes, surface
extraction, enclosed surface volume, point/cell mapping, and face area
vectors.

Backend model
-------------

The project supports two backends:

- ``torch``
- ``phlower_tensor``

The backend is selected when a mesh is created. Internally, the project keeps
geometry and aggregation logic close to torch semantics so that the same high
level API can work across both backends.

Caching model
-------------

The library intentionally treats geometry and topology differently:

- geometry values are recomputed from point coordinates
- topology skeletons are built once and reused
- backend-specific sparse tensors are materialized lazily from cached skeletons

This is the core trade-off behind the project: keep coordinate-dependent values
fresh while avoiding repeated connectivity work.

Terminology
-----------

The documentation uses a small set of consistent abbreviations:

- ``P``: point
- ``C``: cell
- ``F``: face
- ``CP``: cell-point incidence
- ``PP``: point adjacency
- ``CC``: cell adjacency

If you need concrete end-to-end examples, continue with
:doc:`examples_and_api`.
