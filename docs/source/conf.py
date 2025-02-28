# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path().resolve()))

# -- pyvista configuration ---------------------------------------------------
import pyvista

pyvista.start_xvfb()
pyvista.BUILDING_GALLERY = True
pyvista.OFF_SCREEN = True
# Preferred plotting style for documentation
pyvista.set_plot_theme("document")
pyvista.global_theme.window_size = [1024, 768]
pyvista.global_theme.font.size = 22
pyvista.global_theme.font.label_size = 22
pyvista.global_theme.font.title_size = 22
pyvista.global_theme.return_cpos = False
pyvista.set_jupyter_backend(None)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import graphlow

project = "graphlow"
copyright = "2024, RICOS"
author = "RICOS"
version = graphlow.__version__
release = graphlow.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # google, numpy styleのdocstring対応
    "sphinx.ext.githubpages",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
    "pyvista.ext.plot_directive",
    "pyvista.ext.viewer_directive",
    "sphinx_design",
]

sphinx_gallery_conf = {
    "examples_dirs": "../../tutorials",
    "gallery_dirs": "tutorials",
    "within_subsection_order": "FileNameSortKey",
    "filename_pattern": r"/*\.py",
    "image_scrapers": (
        "matplotlib",
        "pyvista",
    ),
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
html_title = "graphlow"
html_show_search_summary = True
html_favicon = "_static/logo.webp"
html_logo = "_static/logo.webp"

# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "exclude-members": "with_traceback",
    "show-inheritance": False,
}
