Release notes
=============

Changes in graphlow between releases.

0.1.0
--------------

This release replaces the pre-0.1 layout with a new package structure and public
surface. **All code targeting 0.0.x must be ported.**

For a **legacy mesh API →** :class:`~graphlow.core.mesh.TensorMesh` method map
(``compute_*``, ``dict_*_tensor``, ``extract_*``, etc.), see
:doc:`user_guide/migration_0_1`.

Removed modules and types
^^^^^^^^^^^^^^^^^^^^^^^^^

- ``graphlow.base`` (including ``GraphlowMesh`` and the former mesh interface).
- ``graphlow.processors`` (``geometry_processor``, ``graph_processor``,
  ``isoAM_processor`` and their workflows).
- ``graphlow.util`` (constants, logger, phlower helpers, sparse helpers, enums).
- ``graphlow.io.io`` (legacy ``read`` implementation path).

New layout (high level)
^^^^^^^^^^^^^^^^^^^^^^^

- ``graphlow.core``: ``TensorMesh``, topology/geometry facades, backends
  (``torch`` / ``phlower``), caching, blocks, face registry.
- ``graphlow.geometry``: differentiable geometric operators (surface, volume,
  distance, operators, etc.).
- ``graphlow.graph``: skeleton adjacency, derived sparse matrices, mapping.
- ``graphlow.io``: PyVista bridge (e.g. ``from_pyvista``).
- ``graphlow.api``: public IO helpers such as :func:`graphlow.read`.
- ``graphlow.utils``: shared enums, topology helpers, logging configuration.

Public imports
^^^^^^^^^^^^^^

Prefer the package root:

.. code-block:: python

   from graphlow import read, from_pyvista, TensorMesh, configure_logging

:func:`~graphlow.read` now builds a :class:`~graphlow.core.mesh.TensorMesh` and
requires an explicit or default ``backend`` (``"torch"`` or ``"phlower"``); see
API reference for parameters.

Documentation and examples
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Sphinx docs reorganized (user guide, contributor guide, API reference).
- Tutorials moved under ``examples/`` (Sphinx-Gallery).

Tests
^^^^^

- Tests are grouped under ``tests/unit/``, ``tests/e2e/``, ``tests/benchmark/``,
  ``tests/profile/``, and ``tests/visualize/``; default ``pytest`` options skip
  benchmark, profile, and slow markers unless opted in.

0.0.2
--------------

- Migrated project management from Poetry to uv (`pyproject.toml`, CI, and local workflow).
- Switched documentation theme to PyData Sphinx Theme and updated layout configuration.

0.0.1
--------------

- First release
