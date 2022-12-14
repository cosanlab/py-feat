# How to install Py-Feat

## Basic setup
You can either install the latest stable version from Pypi using:

```
pip install py-feat
```

Or get the latest development version by installing directly from github:

```
pip install git+https://github.com/cosanlab/py-feat.git
```

```{note}
Py-Feat currently supports both CPU and GPU processing on NVIDIA cards. We have **experimental** support for GPUs on macOS which you can try with `device='auto'`. However, we currently advise using the default (`cpu`) on macOS until PyTorch support stabilizes.
```

## Using Google Colab
On any page in these docs, you can you can simply click on [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() badges you see to open that page in [Google Colab](http://colab.research.google.com/).
```{note}
Make sure to add and run `!pip install py-feat` as the **first** cell into the Colab notebook before running any other code! 
```

## Development setup

If you plan on contributing to py-feat then you might want to install an editable version so that any changes you make to source files are automatically reflected:

```
git clone https://github.com/cosanlab/py-feat.git  
cd py-feat 
pip install -e .
pip install -r requirements-dev.txt
```
