# General contributions guidelines

We always welcome contributions to Py-Feat and recommend you follow these basic steps to do:

1. Fork the repository on [GitHub](https://github.com/cosanlab/feat). 
2. Install Feat on your machine, by `git clone` your fork
3. Install the development dependencies which will *also* install the package dependencies: `pip install -r requirements-dev.txt`
4. Install Py-Feat in development mode so that any changes you make to source files are *automatically* reflected in scripts/notebooks: `pip install -e .`
5. Add you code contributions and/or changes
6. Check or format your code using [black](https://black.readthedocs.io/en/stable/)
7. Create or update the appropriate tests in `feat/tests/`.
8. Run a single test to make sure your new functionality works: `pytest -k 'name_of_your_test'`
9. Alternatively (or additionally) run the full Py-Feat test suite: `pytest`
10. When your tests pass create a pull-request against the `master`/`main` branch on github!

## Tutorial contribution guidelines

All Py-Feat tutorial are made using [jupyter book](https://jupyterbook.org/intro.html). To add a new tutorial or page takes just 3 steps:
1. Add a jupyter notebook or markdown file to `notebooks/content/`
2. Add an entry for your file to the table-of-contents in `notebooks/_toc.yml`
3. Run `jupyter-book build notebooks` to render the documentation

You can check the build jupyter book by opening `notebooks/_build/html/index.html` in your browser.

```{note}
Our documentation building pipeline does **not** execute jupyter notebooks. It just renders their input and output as pages. So make sure you locally execute cells that you want output for **before** committing your changes
```

For instructions on how to add new detectors see [here](./modelContribution.md)