# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from datetime import date
import subprocess
import os

# Doxygen
if not os.path.exists( "_build/xml/" ):
    subprocess.call( 'doxygen ../Doxyfile.in', 
                     shell=True )

project = 'Machine Learning Compilers'
copyright = f'{date.today().year}, Lucas Obitz, Luca-Philipp Grumbach'
author = 'Lucas Obitz, Luca-Philipp Grumbach'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx_copybutton',
    'breathe'
]

templates_path = [
    '_templates'
]
exclude_patterns = [
    '_build',
    '.DS_Store'
]

highlight_language = 'c++'

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',  #  Provided by Google in your dashboard
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    
    'logo_only': False,

    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

html_static_path = [
    '_static'
]

# -- Breathe configuration -------------------------------------------------

breathe_projects = {
    "Machine Learning Compilers": "_build/xml/",
}

breathe_default_project = "Machine Learning Compilers"
breathe_default_members = (
    'members', 
    'undoc-members'
)
