"""
Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html
"""

import os
import sys
from datetime import datetime
import warnings
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass

# Add the project root directory to the Python path (robust to CWD)
_DOCS_DIR = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(_DOCS_DIR, '..')))

# -- Project information -----------------------------------------------------
project = 'CausalKit'
copyright = f'{datetime.now().year}, CausalKit Team'
author = 'Ioann Martynov'

# Version information
try:
    import causalkit as _ck
    release = str(getattr(_ck, "__version__", "0.0.0"))
    version = release
except Exception:
    # Fallback to a placeholder version if import fails in docs env
    release = version = "0.0.0"

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_nb',
    'nbsphinx',
]

# MyST parser settings
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'replacements',
    'smartquotes',
    'substitution',
    'tasklist',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Autodoc settings
# Allow autosummary to generate stubs for API pages that declare them.
# With autodoc_mock_imports below, this should be safe in docs environments.
autosummary_generate = True
# Mock optional heavy dependencies to allow docs build in minimal CI environments
autodoc_mock_imports = [
    'catboost',
    'matplotlib',
    'econml',
    'shap',
]
# Move type hints into the description for clearer parameter sections
autodoc_typehints = "description"
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# The suffix(es) of source filenames.
source_suffix = ['.rst', '.md', '.ipynb']

nbsphinx_thumbnails = {
'research/dgp_benchmarking': '_static/dgp_benchmarking.png',
}

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_theme_options = {
    'logo': {
        'text': 'CausalKit',
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/IoannMartynov/CausalKit",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        }],
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


html_css_files = [
    'css/custom.css',
]

# Favicon for the site
html_favicon = '_static/logo.svg'

# Logo in the navbar (PyData Sphinx Theme)
html_logo = '_static/logo_big.svg'

# Disable "Show Source" links across all pages
html_show_sourcelink = False
html_copy_source = False

# -- Options for nbsphinx output ---------------------------------------------
nbsphinx_execute = 'never'
nbsphinx_allow_errors = False
nbsphinx_timeout = 60

# -- Options for intersphinx extension ---------------------------------------
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
}

# -- Options for linkcheck builder -------------------------------------------
linkcheck_ignore = [r'http://localhost:\d+/']

# -- Options for MyST-NB (myst_nb) --------------------------------------------
# Prevent executing notebooks during Sphinx builds; render existing outputs only.
# Use both legacy and new config keys for compatibility across myst-nb versions.
jupyter_execute_notebooks = "off"  # legacy key used by older myst-nb
nb_execution_mode = "off"           # new key used by myst-nb >= 0.17/1.0
# Be explicit about not raising on error if execution is ever triggered by mistake.
nb_execution_raise_on_error = False