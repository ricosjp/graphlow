Lint and style
==============

The project keeps style enforcement lightweight: use Ruff for code checks and a
small set of documentation conventions for public APIs.

Lint commands
-------------

Run the repository lint target::

   make lint

This currently executes:

- ``uv run ruff check --output-format=full``
- ``uv run ruff format --diff``

The formatting command is run in diff mode, so it shows the changes Ruff would
make without rewriting files automatically.

Code style expectations
-----------------------

At minimum, contributors should keep these points in mind:

- follow existing naming and module boundaries instead of introducing a new
  local style
- keep geometry logic and topology logic separated when possible
- prefer changes that fit the current backend model instead of adding a new
  abstraction layer too early

Docstring conventions
---------------------

Public API docstrings use the NumPy style. The detailed project-specific rules
for return values, shapes, dimension symbols, and sparse array descriptions are
documented in :doc:`docstring_style`.

.. toctree::
   :maxdepth: 1

   docstring_style

When docs need updating
-----------------------

Update documentation when you change:

- public function or method behavior
- argument semantics
- return shapes
- user-facing examples

After doc or example changes, also run :doc:`building_docs`.
