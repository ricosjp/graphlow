Building docs
=============

The documentation is built with Sphinx from ``docs/source`` into ``docs/build``.

Build command
-------------

Use the repository target::

   make document

This target removes the previous HTML output, clears generated API stubs, and
then runs:

.. code-block:: bash

   uv run sphinx-build docs/source docs/build -b html

What the build regenerates
--------------------------

The docs build refreshes several generated artifacts:

- ``docs/build`` for HTML output
- ``docs/source/api_reference/generated`` for autosummary pages
- ``docs/source/example_gallery/auto_examples`` for gallery output

The Makefile also removes ``docs/source/sg_execution_times.rst`` before the
next build so stale timing pages do not linger.

Headless environments
---------------------

PyVista-based gallery rendering may require a virtual display in CI or other
headless environments. The repository documents this workflow as::

   xvfb-run make document

This is especially relevant when examples trigger rendering through
Sphinx-Gallery.

When you should rebuild docs
----------------------------

Rebuild the docs after changing:

- public APIs or docstrings
- pages under ``docs/source``
- gallery examples under ``examples``
- autosummary templates or Sphinx configuration

Related files
-------------

- ``docs/source/conf.py`` for Sphinx configuration
- ``docs/source/_templates`` for autosummary templates
- ``examples`` for gallery inputs
