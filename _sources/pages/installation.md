# How to install Py-Feat

Py-Feat requires **Python 3.11+** (3.11, 3.12, and 3.13 are tested).

## Basic setup

We recommend [**uv**](https://docs.astral.sh/uv/) — a fast, modern Python
package manager. Create an environment and install the latest stable release
from PyPI:

```
uv venv --python 3.13        # creates .venv on Python 3.13 (3.11+ supported)
uv pip install py-feat
```

Then either activate the environment (`source .venv/bin/activate`) or run
commands through it with `uv run`, e.g. `uv run python my_script.py`.

To get the latest development version instead, install directly from GitHub:

```
uv pip install "git+https://github.com/cosanlab/py-feat.git"
```

!!! note
    Prefer plain `pip`? Every command above works by dropping `uv`:
    `pip install py-feat` or `pip install "git+https://github.com/cosanlab/py-feat.git"`.

!!! note
    Py-Feat supports CPU and NVIDIA (CUDA) GPUs, and — since v0.7 — Apple Silicon
    GPUs via Metal (MPS). Pass `device='auto'` (or `device='mps'` / `device='cuda'`)
    when constructing a `Detectorv1` to use the GPU; the default is `cpu`.

## Running tutorials online (molab)
Every tutorial page has an **Open in molab** button at the top that opens the
notebook in [molab](https://molab.marimo.io), marimo's free hosted runtime — run
the tutorials in your browser with no local install required.

## Development setup

To contribute to py-feat, install an editable copy so your source changes are
picked up immediately:

```
git clone https://github.com/cosanlab/py-feat.git
cd py-feat
uv venv
uv pip install -e .
uv pip install -r requirements-dev.txt
```

Run the test suite with `uv run pytest` (use `-m "not network"` to skip tests
that download models from the HuggingFace Hub).
