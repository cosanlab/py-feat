#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = {}
with open("feat/version.py") as f:
    exec(f.read(), version)

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='feat',
    version=version['__version__'],
    description="Facial Expression Analysis Toolbox",
    long_description="",
    author="Jin Hyun Cheong, TianKang Xie, Sophie Byrne, Nathaniel Hanes, Luke Chang ",
    author_email='jcheong0428@gmail.com',
    url='https://github.com/cosanlab/feat',
    packages=find_packages(include=['feat', 'bin/*']),
    package_data = {'feat': ['resources/*','tests/*','tests/data/*']},
    scripts=['bin/detect_fex.py', 'bin/download_models.py'],
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords=['feat','face','facial expression','emotion'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='feat/tests',
    **extra_setuptools_args
)

def _post_install():
    from feat.utils import get_resource_path
    import wget
    import os

    print("Downloading FEX emotion model.")
    fex_emotion_model = "https://github.com/cosanlab/feat/releases/download/v0.1/fer_aug_model.h5"
    wget.download(fex_emotion_model, get_resource_path())

    if os.path.exists(os.path.join(get_resource_path(), "fer_aug_model.h5")):
        print("\nFEX emotion model downloaded successfully.\n")
    else:
        print("Something went wrong. Model not found in directory.")

    print("Downloading landmark detection model.")
    lbfmodel = "https://github.com/cosanlab/feat/releases/download/v0.1/lbfmodel.yaml"
    wget.download(lbfmodel, get_resource_path())

    if os.path.exists(os.path.join(get_resource_path(), "lbfmodel.yaml")):
        print("\nLandmark detection model downloaded successfully.\n")
    else:
        print("Something went wrong. Model not found in directory.")

_post_install()
