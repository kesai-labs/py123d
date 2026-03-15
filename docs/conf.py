# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


from autoclasstoc import PublicDataAttrs


class PublicDataAttrsNoEnum(PublicDataAttrs):
    """Public data attributes, excluding inherited enum attrs ``name`` and ``value``."""

    key = "public-attrs-no-enum"
    exclude_pattern = ["name", "value"]


project = "py123d"
copyright = "2026"
author = "DanielDauner"
release = "v0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "autoclasstoc",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinxcontrib.youtube",
    "sphinx_design",
    "myst_parser",
]

intersphinx_mapping = {
    "rtd": ("https://docs.readthedocs.io/en/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for EPUB output
epub_show_urls = "footnote"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"
html_title = ""

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_favicon = "_static/123D_favicon.svg"

html_theme_options = {}


autodoc_typehints = "both"
autodoc_class_signature = "separated"
autodoc_default_options = {
    "members": True,
    "special-members": False,
    "private-members": False,
    "inherited-members": True,
    "undoc-members": True,
    "member-order": "bysource",
    "exclude-members": "__post_init__, __new__, __weakref__, __iter__,  __hash__, annotations, _array, name, value",
    "imported-members": True,
}

autosummary_generate = True

autoclasstoc_sections = [
    "public-attrs-no-enum",
    "public-methods-without-dunders",
    "private-methods",
]
html_css_files = ["css/theme_overrides.css", "css/version_switch.css"]
html_js_files = ["js/version_switch.js"]


# Custom CSS for color theming
html_css_files = [
    "custom.css",
]

# Additional theme options for color customization
html_theme_options.update(
    {
        "light_logo": "123D_logo_transparent_black.svg",
        "dark_logo": "123D_logo_transparent_white.svg",
        "sidebar_hide_name": True,
        "footer_icons": [
            {
                "name": "GitHub",
                "url": "https://github.com/autonomousvision/py123d",
                "html": """
            <svg stroke="currentColor" fill="currentColor" stroke-width="0" viewBox="0 0 16 16">
                <path fill-rule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z"></path>
            </svg>
        """,
                "class": "",
            },
        ],
    }
)

html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "sidebar/variant-selector.html",
    ]
}

# This CSS should go in /home/daniel/py123d_workspace/py123d/docs/_static/custom.css
# Your conf.py already references it in html_css_files = ["custom.css"]

# If you want to add custom CSS via configuration, you can use:
html_css_files = ["custom.css"]
