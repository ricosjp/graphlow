Quickstart
==========

This page shows the shortest path from a mesh file to a usable ``TensorMesh``.

Load a mesh from a file
-----------------------

Use :func:`graphlow.read` when your input is a path on disk.

.. code-block:: python

   import pathlib
   import graphlow

   file = pathlib.Path("path/to/mesh.vtu")
   mesh = graphlow.read(file, backend="torch")

Load a mesh from PyVista
------------------------

Use :func:`graphlow.from_pyvista` when you already have a PyVista mesh in
memory.

.. code-block:: python

   import pyvista as pv
   import graphlow

   pv_mesh = pv.read("path/to/mesh.vtu")
   mesh = graphlow.from_pyvista(pv_mesh, backend="torch")

Run a few common operations
---------------------------

Once loaded, the mesh exposes geometry and topology helpers.

.. code-block:: python

   volumes = mesh.geometry.cell_volumes()
   centroids = mesh.geometry.cell_centroids()
   adjacency = mesh.topology.point_adjacency()

For exact public method names and signatures, see the
:doc:`/api_reference/index`.

Use autograd on mesh coordinates
--------------------------------

``graphlow`` is designed for differentiable geometry workflows.

.. code-block:: python

   import graphlow

   mesh = graphlow.read("path/to/mesh.vtu", backend="torch")
   mesh.requires_grad(True)

   loss = mesh.geometry.cell_volumes().sum()
   loss.backward()

What to read next
-----------------

- Go to :doc:`core_concepts` to understand how ``TensorMesh``, geometry, and
  topology fit together.
- Go to :doc:`examples_and_api` when you want longer tutorials or exact API
  signatures.
