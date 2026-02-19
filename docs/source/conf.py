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
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx.ext.githubpages",
    "sphinxcontrib.mermaid",
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
# https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # Logo configuration
    "logo": {
        "image_light": "_static/logo.webp",
        "image_dark": "_static/logo.webp",
    },
    # Navbar configuration
    "navbar_start": ["navbar-logo", "navbar-version"],
    "navbar_align": "content",
    "header_links_before_dropdown": 5,
    # Right-hand sidebar contents
    "secondary_sidebar_items": ["page-toc"],
    # Footer configuration (hide theme/version credits)
    "footer_start": ["copyright"],
    "footer_end": [],
    "footer_center": [],
}
html_context = {
    "github_version": "main",
    "doc_path": "docs/source/",
    "default_mode": "light",
}
html_sidebars: dict[str, list] = {"index": []}
html_static_path = ["_static"]
html_title = "graphlow"
html_show_search_summary = True
html_favicon = "_static/logo.webp"
html_logo = "_static/logo.webp"
html_show_sphinx = False
html_css_files = ["custom.css"]


# -- Extension configuration -------------------------------------------------
autosummary_generate = True
autodoc_typehints = "description"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "exclude-members": "with_traceback",
    "show-inheritance": False,
}

# numpydoc configuration
numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True
# Validation checks: empty set to disable, or list of check codes to enable
# Common checks: GL01 (line too long), EX01 (examples not found), etc.
numpydoc_validation_checks = set()  # Disable validation for now
numpydoc_validation_exclude = set()  # Exclude specific checks if needed

# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]

# codeautolink
codeautolink_autodoc_inject = False
codeautolink_search_css_classes = ["highlight-default"]
codeautolink_concat_default = True
