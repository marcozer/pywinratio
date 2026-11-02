"""Sphinx configuration for the pywinratio documentation."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# notebooks use the inline matplotlib backend; never force MPLBACKEND here
os.environ.pop("MPLBACKEND", None)

import pywinratio  # noqa: E402

project = "pywinratio"
author = "Marc-Anthony Chouillard and Thomas Husson"
copyright = "2026, Marc-Anthony Chouillard"
release = pywinratio.__version__
version = release

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
]

source_suffix = {".rst": "restructuredtext", ".md": "myst-nb"}
exclude_patterns = ["_build", "jupyter_execute", "**.ipynb_checkpoints"]
templates_path = ["_templates"]

# MyST / notebooks: every tutorial is executed at build time
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist", "attrs_inline"]
myst_heading_anchors = 3
nb_execution_mode = os.environ.get("NB_EXECUTION_MODE", "force")
nb_execution_raise_on_error = True
nb_execution_timeout = 900
nb_execution_show_tb = True
nb_merge_streams = True

# API reference
autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False}
autodoc_typehints = "none"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False
numpydoc_attributes_as_param_list = False

intersphinx_timeout = 30
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
}

html_theme = "pydata_sphinx_theme"
html_title = f"pywinratio {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "github_url": "https://github.com/marcozer/pywinratio",
    "navbar_align": "left",
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
}
html_context = {
    "github_user": "marcozer",
    "github_repo": "pywinratio",
    "github_version": "main",
    "doc_path": "docs",
}
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
