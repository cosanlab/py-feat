#!/usr/bin/env python
# -*- coding: utf-8 -*-

import atexit, os, sys
from setuptools import setup, find_packages
from setuptools.command.install import install

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = {}
with open("feat/version.py") as f:
    exec(f.read(), version)

extra_setuptools_args = dict(
    tests_require=['pytest']
)

setup(
    name='py-feat',
    version=version['__version__'],
    description="Facial Expression Analysis Toolbox",
    long_description="",
    author="Jin Hyun Cheong, Tiankang Xie, Sophie Byrne, Nathaniel Hanes, Luke Chang ",
    author_email='jcheong0428@gmail.com',
    url='https://github.com/cosanlab/feat',
    packages=find_packages(),
    package_data = {'feat': ['resources/*','tests/*','tests/data/*']},
    scripts=['bin/detect_fex.py', 'bin/download_models.py'],
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords=['feat','face','facial expression','emotion'],
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='feat/tests',
    **extra_setuptools_args
)


