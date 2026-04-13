Architecture
============

This page summarizes the parts of the design note that matter most when you are
trying to modify the codebase safely.

Design goals
------------

The project is built around three goals:

1. compute differentiable geometry metrics from mesh coordinates
2. build topology-derived graph and sparse operators
3. expose the same workflow through torch and phlower_tensor backends

Core design split
-----------------

The most important boundary in the project is this:

- coordinate-dependent geometry should stay differentiable
- connectivity-derived topology should stay cacheable

In practice, this means geometry values are recomputed from points while
topology skeletons and backend sparse materializations are reused.

High-level module map
---------------------

.. mermaid::

   flowchart TD
       tensorMesh["TensorMesh"] --> geometryLayer["mesh.geometry"]
       tensorMesh --> topologyLayer["mesh.topology"]
       tensorMesh --> pyvistaMesh["PyVista mesh"]
       geometryLayer --> geometryPkg["src/graphlow/geometry"]
       topologyLayer --> graphPkg["src/graphlow/graph"]
       tensorMesh --> backendLayer["Backend / BackendCache"]

Main directories
----------------

- ``src/graphlow/core``: ``TensorMesh`` and wrapper objects for geometry and
  topology
- ``src/graphlow/geometry``: geometry metrics, operators, and analytic or
  isoparametric methods
- ``src/graphlow/graph``: skeleton builders, mappings, and sparse graph logic
- ``src/graphlow/io``: user-facing mesh loading entry
  points

Data model
----------

``TensorMesh`` keeps:

- backend tensors for ``points``, ``point_data``, and ``cell_data``
- the original PyVista mesh as the static topology source
- attached geometry and topology helpers

This lets autograd follow coordinate and data tensors without turning the whole
topology pipeline into a differentiable object.

Caching model
-------------

Two cache layers matter:

- a backend-neutral topology cache that stores scipy sparse skeletons and
  blocks
- a backend-specific cache that materializes sparse tensors only when needed

That split is the reason many topology changes should avoid touching geometry
code, and vice versa.

Contributor reading order
-------------------------

If you are new to the codebase, the efficient reading order is:

1. ``src/graphlow/core/mesh.py``
2. ``src/graphlow/core/geometry.py`` and ``src/graphlow/core/topology.py``
3. the concrete implementation modules under ``src/graphlow/geometry`` or
   ``src/graphlow/graph``

For concrete examples of that traversal, continue with :doc:`internals/index`.
