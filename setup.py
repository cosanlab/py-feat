#!/usr/bin/env python
# -*- coding: utf-8 -*-

import atexit, os, sys
from setuptools import setup, find_packages
from setuptools.command.install import install
import zipfile

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

version = {}
with open("feat/version.py") as f:
    exec(f.read(), version)

extra_setuptools_args = dict(
    tests_require=['pytest']
)

class CustomInstall(install):
    def run(self):
        def _post_install():
            from torchvision.datasets.utils import download_url
            import os, sys, json

            def get_resource_path():
                for p in sys.path:
                    if os.path.isdir(p) and "feat" in os.listdir(p):
                        return os.path.join(p, "feat", "resources")

            with open(os.path.join(get_resource_path(), "model_list.json"), "r") as f:
                model_urls = json.load(f) 

            # Download default models
            default_models = [("au_detectors", "rf"), ("emotion_detectors", "resmasknet"), ("face_detectors", "retinaface"), ("landmark_detectors", "mobilenet")]   
            for modelType, modelName in default_models:
                for url in model_urls[modelType][modelName]["urls"]:
                    download_url(url, get_resource_path())

        atexit.register(_post_install)
        install.run(self)

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
    cmdclass={'install':CustomInstall},
    **extra_setuptools_args
)


