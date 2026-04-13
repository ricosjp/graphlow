Installation
============

``graphlow`` supports a default torch backend and an optional
phlower_tensor backend.

Install with ``uv`` (recommended)
---------------------------------

Install with the CPU Torch backend::

   uv add graphlow[cpu]

Install with a CUDA backend (example: CUDA 12.4)::

   uv add graphlow[cu124]

Install with the optional phlower backend (combine as needed)::

   uv add graphlow[cpu,phlower]
   uv add graphlow[cu124,phlower]

Install with ``pip``
--------------------

If your environment does not use ``uv``, the same package layout is available
through ``pip``::

   pip install graphlow[cpu]
   pip install graphlow[cu124]
   pip install graphlow[cpu,phlower]

Requirements
------------

- Python 3.12 or newer
- One Torch backend extra selected: ``cpu``, ``cu118``, or ``cu124``
- PyVista-compatible mesh input such as ``.vtu``, ``.vtp``, or other VTK-based
  files handled by PyVista

Choosing a backend
------------------

Use ``backend="torch"`` for PyTorch-based execution (after installing one of
``cpu``, ``cu118``, or ``cu124``). Use ``backend="phlower"`` when your
workflow depends on ``phlower_tensor`` semantics.

The backend is chosen when creating a mesh object:

.. code-block:: python

   import graphlow

   mesh = graphlow.read("path/to/mesh.vtu", backend="torch")

Next step
---------

Continue with :doc:`quickstart` to load your first mesh and inspect the object
model.
