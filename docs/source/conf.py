# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
from pathlib import Path

project = "faenet"
copyright = "2023, OCP"
author = "Victor Schmidt"

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration
root_doc = "index"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "sphinx_autodoc_typehints",
    "sphinx.ext.inheritance_diagram",
    "sphinx.ext.napoleon",
    "sphinx_rtd_theme",
    "sphinx_math_dollar",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx_design",
    "sphinx_copybutton",
]

# sphinx.ext.intersphinx
intersphinx_mapping = {
    "torch": ("https://pytorch.org/docs/stable", None),
}
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}


# sphinx.ext.autodoc & autoapi.extension
# https://autoapi.readthedocs.io/
autosummary_generate = True
autodoc_typehints = "description"
autoapi_type = "python"
autoapi_dirs = [str(ROOT / "faenet")]
autoapi_member_order = "groupwise"
autoapi_template_dir = "_templates/autoapi"
autoapi_python_class_content = "class"
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    # "imported-members",
    "special-members",
]
autoapi_keep_files = False
autoapi_root = "api"
autoapi_add_toctree_entry = False

mathjax_path = (
    "https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
)
mathjax3_config = {
    "tex": {
        "inlineMath": [
            ["$", "$"],
            ["\\(", "\\)"],
        ],
        "processEscapes": True,
    },
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_static_path = ["_static"]

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
}
