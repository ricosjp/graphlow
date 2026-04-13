Overview
========

``graphlow`` is a differentiable mesh-graph library built around one core
idea: geometry and topology should be available through a single mesh object
while still working with autograd-friendly tensor backends.

What contributors usually touch
-------------------------------

Depending on the change, you will usually work in one or more of these areas:

- ``src/graphlow/core`` for the mesh container, topology wrapper, and geometry
  wrapper
- ``src/graphlow/geometry`` for coordinate-dependent computations
- ``src/graphlow/graph`` for sparse topology builders and mapping operators
- ``tests`` for unit and end-to-end coverage
- ``docs/source`` and ``examples`` for Sphinx documentation and gallery content

Contributor workflow in practice
--------------------------------

For most changes, the shortest reliable loop is:

1. set up the development environment with ``uv``
2. make the code change
3. run the relevant tests
4. run lint checks
5. rebuild the Sphinx docs if you touched public APIs, examples, or docs

Read this guide in that same order:

1. :doc:`development_setup`
2. :doc:`testing`
3. :doc:`lint_and_style`
4. :doc:`building_docs`
5. :doc:`architecture`

What this guide does not invent
-------------------------------

This guide intentionally focuses on what is already defined in the repository.
It does not guess a pull request template, branching model, or release process
that has not been documented yet.
