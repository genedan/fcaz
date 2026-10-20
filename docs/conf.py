import os
import sys

sys.path.insert(
    0,
    os.path.abspath('..'),
)

from fcaz.constants import BUILD_VERSION

project = 'fcaz'
author = 'Gene Dan'
copyright = '2026, Gene Dan'
release = version = BUILD_VERSION

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
]

autosummary_generate = True
napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'scipy': ('https://docs.scipy.org/doc/scipy', None),
}

exclude_patterns = ['_build']
html_theme = 'pydata_sphinx_theme'
html_baseurl = 'https://casact.github.io/fcaz/'
html_theme_options = {
    'github_url': 'https://github.com/casact/fcaz',
}
html_static_path = ['_static']
