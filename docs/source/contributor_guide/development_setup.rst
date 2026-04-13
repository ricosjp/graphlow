Development setup
=================

The project uses ``uv`` for dependency management and local command execution.

Python version
--------------

``graphlow`` requires Python 3.12 or newer.

Install dependencies
--------------------

For contributor work, install the development groups with an explicit Torch
backend extra::

   uv sync --refresh --reinstall --extra cu124 --extra phlower --group dev

This matches the ``dev-install`` target in the project ``Makefile`` and brings
in test, lint, docs, visualization, and notebook dependencies.

Useful Make targets
-------------------

The repository already defines the common local commands:

- ``make install``: refresh and reinstall the default dependency set
- ``make dev-install``: install all development dependencies
- ``make lint``: run Ruff checks and formatting diff
- ``make cpu-test``: run the default test suite with coverage
- ``make document``: rebuild the Sphinx HTML output

Editable source layout
----------------------

The package source lives under ``src/graphlow``. The documentation assumes this
layout and Sphinx inserts ``src`` into ``sys.path`` from ``docs/source/conf.py``.

When to use the dev environment
-------------------------------

Use the full development environment if you are doing any of the following:

- adding or changing tests
- touching Sphinx docs or example gallery content
- changing public APIs
- working on performance tooling or visualization helpers

Next step
---------

Go to :doc:`testing` after setup so your first local feedback loop is reliable.
