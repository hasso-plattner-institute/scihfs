import pathlib
import sys

import sphinx_rtd_theme

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

# -- Path setup --------------------------------------------------------------

# Add the project root (one level up from /doc) to sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

# -- Project metadata --------------------------------------------------------

with open("../pyproject.toml", "rb") as f:
    pyproject = tomllib.load(f)

project = pyproject["project"]["name"]
author = author = ", ".join(a["name"] for a in pyproject["project"].get("authors", []))
version = release = pyproject["project"]["version"]

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "sphinx_gallery.gen_gallery",
]

autodoc_typehints = "description"
autodoc_member_order = "bysource"

templates_path = ["_templates"]
exclude_patterns = ["_build", "_templates", "_static"]

# -- Napoleon settings -------------------------------------------------------

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_static_path = ["_static"]
html_style = "css/project-template.css"

# -- BibTeX  -----------------------------------------------------------------

bibtex_bibfiles = ["refs.bib"]

# -- Sphinx-Gallery -------------------------------------------------------------

sphinx_gallery_conf = {
    "examples_dirs": ["../examples"],
    "gallery_dirs": ["auto_examples"],
}
