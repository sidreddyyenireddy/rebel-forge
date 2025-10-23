import skala

project = "Skala"
version = skala.__version__
author = "Microsoft Research, AI for Science"

extensions = [
    "myst_nb",
    "sphinx_book_theme",
    "sphinx.ext.autodoc",
]

nb_execution_timeout = 300  # 5 minutes, set to -1 for no timeout
nb_merge_streams = True  # Merge multiple outputs from the same cell into one box

html_theme = "sphinx_book_theme"
html_title = project
html_logo = "_static/img/density.png"
html_favicon = "_static/img/density.png"
master_doc = "index"
