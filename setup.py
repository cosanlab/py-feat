#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='feat',
    version='0.0.1',
    description="Facial Expression Analysis Toolbox",
    long_description="",
    author="Jin Hyun Cheong, Nathaniel Hanes, Luke Chang ",
    author_email='jcheong0428@gmail.com',
    url='https://github.com/cosanlab/feat',
    packages=find_packages(include=['feat']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='feat',
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
    test_suite='tests'
)
