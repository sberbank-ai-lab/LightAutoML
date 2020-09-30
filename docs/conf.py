# Configuration file for the Sphinx documentation builder.
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
import sphinx
import datetime
from docutils.parsers.rst import Directive


CURR_PATH = os.path.abspath(os.path.dirname(__file__))
LIB_PATH = os.path.join(CURR_PATH, os.path.pardir, 'lightautoml')
sys.path.insert(0, LIB_PATH)


project = 'LightAutoML'
copyright = '%s, Sberbank AI Lab' % str(datetime.datetime.now().year)
author = 'Sberbank AI Lab'


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',  # will be used for tables
    'sphinx.ext.viewcode',  # for [source] button
    'sphinx.ext.napoleon' # structure
]

exclude_patterns = [
    'build/*',
    '_book/*'
]

# Delete external references
# autodoc_mock_imports = ['numpy', 'pandas']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# code style
pygments_style = 'default'

# autodoc
# function names that will not be included in documentation
EXCLUDED_MEMBERS = ','.join(['get_own_record_history_wrapper',
                             'get_record_history_wrapper',
                             'record_history_omit',
                             'record_history_only'])

# function names that will be included in documentation by force
SPECIAL_MEMBERS = ','.join(['__init__', '__repr__',
                            '__len__', '__getitem__',
                            '__setitem__', '__iter__',
                            '__next__'])

autodoc_default_options = {
    'ignore-module-all': True,
    'undoc-members': False,
    'show-inheritance': True,
    'special-members': SPECIAL_MEMBERS,
    'exclude-members': EXCLUDED_MEMBERS
}

# order of members in docs, usefully for methods in class
autodoc_member_order = 'bysource'

# typing, use in signature
autodoc_typehints = 'signature'

# to omit some __init__ methods in classes where it not defined
autoclass_content = 'class'

# all warnings will be produced as errors
autodoc_warningiserror = True

# when there is a link to function not use parentheses
add_function_parentheses = False

# napoleon
# in this docs google docstring format used
napoleon_google_docstring = True
napoleon_numpy_docstring = False

napoleon_include_init_with_doc = False

# to omit private members
napoleon_include_private_with_doc = False

# use spectial members
napoleon_include_special_with_doc = True

napoleon_use_param = True

# True to use a :keyword: role for each function keyword argument
napoleon_use_keyword = True

# True to use the .. admonition:: directive for References sections instead .. rubric::
napoleon_use_admonition_for_examples = True


def setup(app):
    app.add_css_file('style.css')  # customizing default theme
