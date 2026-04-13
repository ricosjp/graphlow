"""
Sphinx configuration for graphlow.

Docstrings/comments in this file are intentionally kept minimal and organized
in sections for readability.
"""

from __future__ import annotations

import pathlib
import sys
from sphinx_gallery.sorting import ExplicitOrder

# -- Paths -------------------------------------------------------------------
# Repository root (conf.py lives in docs/source/).
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS_SOURCE_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(DOCS_SOURCE_DIR))
sys.path.insert(0, str(REPO_ROOT / "src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
from importlib.metadata import version as _version

project = "graphlow"
author = "RICOS"
copyright = "2024, RICOS"
version = release = _version("graphlow")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
extensions = [
    # Core
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    # Docstring style
    "numpydoc",
    # Gallery
    "sphinx_gallery.gen_gallery",
    # UX
    "sphinx_copybutton",
    "sphinx_codeautolink",
    # Diagrams
    "sphinxcontrib.mermaid",
]

templates_path = ["_templates"]
exclude_patterns: list[str] = []

source_suffix = {
    ".rst": "restructuredtext",
}

# Autosummary / Numpydoc -----------------------------------------------------
autosummary_generate = True
autosummary_imported_members = False
numpydoc_show_class_members = False
autodoc_default_options = {
    "members": False,
    "inherited-members": False,
}

# -- Sphinx-Gallery ----------------------------------------------------------
# https://sphinx-gallery.github.io/
sphinx_gallery_conf = {
    # Input examples directory (outside docs/source).
    "examples_dirs": str(REPO_ROOT / "examples"),
    # Output gallery directory (inside docs/source).
    "gallery_dirs": "example_gallery/auto_examples",
    # Render basic examples before advanced ones.
    "subsection_order": ExplicitOrder(
        [
            "../../examples/basic",
            "../../examples/advanced",
        ]
    ),
    # Keep a deliberate teaching order within each section.
    "within_subsection_order": ExplicitOrder(
        [
            "create_tensor_mesh.py",
            "cell_volumes.py",
            "extract_surface.py",
            "surface_enclosed_volume.py",
            "point_cell_mapping.py",
            "face_area_vectors.py",
            "gradient.py",
            "target_fitting.py",
            "shape_optimization.py",
            "sphere_eversion_continuous_metrics.py",
        ]
    ),
    # Execute all .py examples.
    "filename_pattern": r".*\.py",
    # Route graphlow examples through a wrapper scraper.
    "image_scrapers": ("pyvista",),
    # Avoid cluttering pages with timing.
    "show_memory": False,
}

# -- pyvista (for gallery rendering) -----------------------------------------
# In headless CI (GitHub Actions, GitLab CI, etc.), run the doc build inside a
# virtual display, e.g. ``xvfb-run make document`` (install xvfb in the runner),
# or use VTK built with OSMesa so no display is needed.
import pyvista

pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
try:
    # Optional; avoid hard dependency on IPython for docs builds.
    pyvista.set_jupyter_backend(None)
except ImportError:
    pass

# -- Options for HTML output -------------------------------------------------
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html
html_theme = "pydata_sphinx_theme"
html_title = "graphlow"

html_logo = "_static/logo.webp"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# PyData theme options (minimal defaults; expand as needed).
html_theme_options = {
    "navbar_align": "content",
    "logo": {
        "text": "graphlow",
        "image_light": "_static/logo.webp",
    },
}

# -- Copybutton --------------------------------------------------------------
copybutton_prompt_text = r">>> |\.\.\. "
copybutton_prompt_is_regexp = True

# -- Mermaid -----------------------------------------------------------------
# Render Mermaid in HTML (JS). For non-HTML builders, use mermaid-cli.
mermaid_output_format = "raw"
