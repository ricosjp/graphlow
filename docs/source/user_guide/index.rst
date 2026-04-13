User guide
==========

``graphlow`` is a differentiable mesh-graph library for mesh processing.
It combines geometry metrics, mesh topology, and sparse graph operators behind
one Python API so that the same workflow can be used with torch and
phlower_tensor backends.

This guide is organized by the questions new users usually have first:

.. toctree::
   :maxdepth: 1

   installation
   quickstart
   logging
   core_concepts
   migration_0_1
   examples_and_api

What you can do with ``graphlow``
---------------------------------

Typical workflows include:

- load a PyVista or VTK mesh into a differentiable ``TensorMesh``
- compute geometry metrics such as volumes, normals, centroids, and quality
- build topology-derived sparse operators such as incidences, adjacency, and
  Laplacians
- combine geometry and graph operators in autograd-based optimization loops

If you want the shortest path to a working example, start with
:doc:`quickstart`. If you need package installation details first, go to
:doc:`installation`.
