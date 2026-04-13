Testing
=======

The repository separates fast default tests from optional slow, GPU, benchmark,
and profiling workflows.

Default test behavior
---------------------

The default ``pytest`` configuration excludes the following markers:

- ``benchmark``
- ``profile``
- ``slow``

That means a plain ``uv run pytest`` is expected to cover fast unit and e2e
tests only.

Test layout
-----------

The design note defines two main test layers:

- ``tests/unit``: isolated tests for individual modules
- ``tests/e2e``: stack-level tests that start from ``read`` or
  ``from_pyvista`` and exercise topology or geometry workflows end to end

Marker meanings
---------------

- ``slow``: long-running tests, typically for larger meshes or more expensive
  workflows
- ``benchmark`` and ``profile``: performance investigation targets, not part of
  the normal correctness loop

Device selection
----------------

Tests can switch execution device using ``--device`` (registered in
``tests/conftest.py``):

- ``--device=cpu`` (default)
- ``--device=cuda`` (skips if CUDA is unavailable)

Common commands
---------------

Sync CPU backend and run the fast default suite::

   uv sync --group test --extra cpu
   uv run pytest

Run the Makefile target with coverage::

   make cpu-test

Run tests that include ``slow``::

   uv run pytest -m slow

Sync CUDA backend and run tests on CUDA device (if available)::

   uv sync --group test --extra cu124
   uv run pytest --device=cuda

Run all markers without exclusions::

   uv run pytest -m ""

Related Make targets
--------------------

- ``make cpu-test``
- ``make gpu-test``
- ``make slow-test``

Coverage expectations
---------------------

When you change behavior in ``src/graphlow``, add or update tests in the layer
that matches the change:

- use unit tests for local algorithm behavior
- use e2e tests when the change depends on mesh loading, wrappers, or multiple
  subsystems working together
