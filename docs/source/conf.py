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
import inspect
import importlib
sys.path.insert(0, os.path.abspath('../../src'))


# -- Project information -----------------------------------------------------

project = 'NumbaCS'
copyright = '2025, Albert Jarvis'
author = 'Albert Jarvis'

# The full version, including alpha/beta/rc tags
release = '0.1.2'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
		'sphinx.ext.napoleon',
		'sphinx.ext.autosummary',
		'autoapi.extension',
		'sphinxcontrib.bibtex',
		'sphinx.ext.autosectionlabel',
		'myst_parser',
		'sphinx_gallery.gen_gallery',
		'sphinx.ext.linkcode',
		]

autoapi_type = 'python'
autoapi_dirs = ['../../src/numbacs']
autoapi_options = [
		   'undoc-members',
		   'imported-members',
	  	    ]

bibtex_bibfiles = ['refs.bib']
bibtex_default_style = 'unsrt'

autosectionlabel_prefix_document = True

sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
     'nested_sections': False,
     'within_subsection_order': "FileNameSortKey",
     'matplotlib_animations': True,
}

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

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    'css/custom.css',
]

# -- Linkcode implementation ---------------------------------------------------

def linkcode_resolve(domain, info):
    """
    Determine the URL corresponding to a Python object.
    """
    if domain != 'py' or not info['module']:
        return None

    # Get the Python object being documented
    try:
        obj = importlib.import_module(info['module'])
        for part in info['fullname'].split('.'):
            obj = getattr(obj, part)

        # Unwrap decorators to get the original function
        obj = inspect.unwrap(obj)

        # Get the source file and line numbers
        file = inspect.getsourcefile(obj)
        if file is None:
            return None

        lines, start_line = inspect.getsourcelines(obj)

        # Make the file path relative to the project root
        project_root = os.path.dirname(os.path.abspath('../../'))
        file = os.path.relpath(file, start=project_root)

        try:
            url_path = file[file.rfind('src/'):]
        except ValueError:
            return None

    except (TypeError, AttributeError, ImportError, ValueError, KeyError):
        return None

    # Get the branch/tag/commit from Read the Docs environment variables
    # Fallback to 'main' for local builds
    version = os.environ.get('READTHEDOCS_VERSION', 'main')
    if version == 'latest':
        version = 'main'

    # Construct the GitHub URL
    user = "alb3rtjarvis"
    repo = "numbacs"

    end_line = start_line + len(lines) - 1

    return f"https://github.com/{user}/{repo}/blob/{version}/{url_path}#L{start_line}-L{end_line}"
