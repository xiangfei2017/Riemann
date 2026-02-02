# Configuration file for the Sphinx documentation builder (Chinese version).
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'Riemann'
copyright = '2025, Fei Xiang'
author = 'Fei Xiang'

# The full version, including alpha/beta/rc tags
release = '0.3.0'

# 设置语言为中文
language = 'zh_CN'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# -- Options for autodoc extension ------------------------------------------

# This value selects what content will be inserted into the main body of an autoclass directive.
autoclass_content = 'both'

# This value selects if automatically documented members are sorted alphabetically (value 'alphabetical'), by member type (value 'groupwise') or by source order (value 'bysource').
autodoc_member_order = 'bysource'

# This value controls whether to document special members using autodoc.
autodoc_special_members = '__init__'

# -- Options for intersphinx extension --------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
}

# -- Options for Napoleon extension ------------------------------------------

# Enable parsing of Google style docstrings
napoleon_google_docstring = True

# Enable parsing of NumPy style docstrings
napoleon_numpy_docstring = True

# Whether to include init docstrings
napoleon_include_init_with_doc = False

# Whether to include private members (like _membername) with docstrings
napoleon_include_private_with_doc = False

# Whether to include special members (like __membername__) with docstrings
napoleon_include_special_with_doc = True

# Use the .. admonition:: directive for the Example and Examples sections
napoleon_use_admonition_for_examples = False

# Use the .. admonition:: directive for the Note and Notes sections
napoleon_use_admonition_for_notes = False

# Use the .. admonition:: directive for the References and References sections
napoleon_use_admonition_for_references = False

# Use the :ivar: role for instance variables
napoleon_use_ivar = False

# Use the :param: role for function parameters
napoleon_use_param = True

# Use the :rtype: role for the return type
napoleon_use_rtype = True