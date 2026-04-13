API reference
=============

This reference is generated from the documented Python objects listed below.
For ``mesh.geometry`` and ``mesh.topology``, see ``MeshGeometry`` and
``MeshTopology`` respectively.

Root package functions and enums
--------------------------------

.. autosummary::
   :toctree: generated

   graphlow.read
   graphlow.from_pyvista
   graphlow.FloatPrecision
   graphlow.configure_logging

:func:`graphlow.configure_logging` is an optional helper for quick, interactive
logging setup (notebooks, REPL, short scripts). It does not replace configuring
:mod:`logging` in applications; see :doc:`/user_guide/logging`.

Core mesh classes
-----------------

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst

   graphlow.TensorMesh
   graphlow.core.MeshGeometry
   graphlow.core.MeshTopology
