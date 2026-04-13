Docstring style (Returns and shapes)
=====================================

This project uses NumPy-style docstrings. The following rules apply to the
**Returns** section and to **shape notation** in docstrings.

Returns section
---------------

Single return
~~~~~~~~~~~~~
Use the type only. Do not add a variable name.

- Use ``T`` or the concrete type (e.g. ``TensorMesh[T]``, ``scipy.sparse.csr_array``).
- Example: ``T`` with description "Tensor of shape ``(n_cells, 1)``."

Multiple returns (tuple)
~~~~~~~~~~~~~~~~~~~~~~~~
Always list each return as **name : type**. Use lowercase with underscores,
matching the actual variable names.

- Example: ``isoam : T``, ``moment_inv : T or None``.
- Do not use mixed case for return names (e.g. use ``isoam``, not ``isoAM``).

Shape notation
--------------

- **Always** write shapes as RST inline literals: use double backticks, e.g. ``(n_points, 3)``.
- Phrasing: "shape ``(n_cells, 1)``" or "Tensor of shape ``(n_cells, 1)``."
- For multiple possible shapes: "shape ``(n_faces, 3)`` for volume meshes or ``(n_cells, 3)`` for surface meshes" (each tuple in backticks).

Dimension symbols
-----------------

- Use **lowercase with underscores**: ``n_points``, ``n_cells``, ``n_faces``, ``n_elements``. Do not use ``N_cells`` or ``N_faces``.
- Spatial dimension: use ``dims``.
- Internal notation like ``(K, N, N)``: if kept, add a one-line note e.g. "K = dims, N = n_points" and write the shape in backticks: ``(K, N, N)``.

dtype and layout notes
----------------------

- **Backend tensors**: "Backend tensor of shape ``(n_points, n_points)``." Add "(sparse COO)" or dtype in parentheses if needed.
- **scipy.sparse**: Keep dtype as "dtype: np.int64" on the next line. e.g. "CSR array of shape ``(n_cells, n_points)``. dtype is np.int64.".
