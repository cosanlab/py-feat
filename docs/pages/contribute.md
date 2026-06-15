# General contributions guidelines

We always welcome contributions to Py-Feat and recommend you follow these basic steps to do. We highly recommend using [Visual Studio Code](https://code.visualstudio.com/) as we include recommended editor extensions and settings in this repo. We use `pytest` for testing and `ruff` for linting and formatting:

1. Fork the repository on [GitHub](https://github.com/cosanlab/feat). 
2. Install Feat on your machine, by `git clone` your fork
3. Install the development dependencies which will *also* install the package dependencies: `pip install -r requirements-dev.txt`
4. Install Py-Feat in development mode so that any changes you make to source files are *automatically* reflected in scripts/notebooks: `pip install -e .`
5. Add you code contributions and/or changes
6. Create or update the appropriate tests in `feat/tests/`.
7. Run a single test to make sure your new functionality works: `pytest -k 'name_of_your_test'`
8. Alternatively (or additionally) run the full Py-Feat test suite: `pytest`
9. Add any applicable licenses to `LICENSE.txt`
10. When your tests pass create a pull-request against the `master`/`main` branch on github!

## Tutorial contribution guidelines

Py-Feat's docs are built with **marimo-book** (Material for MkDocs under the hood). To add a new tutorial or page:
1. Add a marimo notebook (`.py`) or a Markdown file under `docs/` — tutorials live in `docs/basic_tutorials/`, prose pages in `docs/pages/`.
2. Add an entry for your file to the table of contents in `docs/book.yml`.
3. Run `cd docs && marimo-book build` to render the site (output in `docs/_site/`).

For executable tutorials marked `mode: cached`, run `marimo-book render` on a machine with the models/data, then commit the refreshed `docs/_rendered/` so CI can build the page without executing it.

!!! note
    Our documentation building pipeline does **not** execute jupyter notebooks. It just renders their input and output as pages. So make sure you locally execute cells that you want output for **before** committing your changes

For instructions on how to add new detectors see [here](./modelContribution.md)