# Notebooks

**Ordinary notebooks (either Jupyter or Mathematica) are disallowed in this repository.**
This decision is motivated by the fact that they are not friendly for Version Control Systems – they store generated outputs and a single modification (e.g., the parameters of an image) can result in thousands of changes.

Hence, we will use replacement for notebooks:

  - For Mathematica we use WolframScript (`*.wls`). For the motivation see [this post](https://mathematica.stackexchange.com/a/155268).
  - For Jupyter Notebooks we use [JupyText](https://github.com/mwouts/jupytext). However, it's perhaps a better idea to manually rewrite the notebook into a script and store it in the `scripts/` directory.
  - For R we use [RMarkDown](https://rmarkdown.rstudio.com/).


