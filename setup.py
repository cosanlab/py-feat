from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

version = {}
with open("feat/version.py") as f:
    exec(f.read(), version)

extra_setuptools_args = dict(tests_require=["pytest"])

setup(
    name="py-feat",
    version=version["__version__"],
    description="Facial Expression Analysis Toolbox",
    long_description="Facial Expression Analysis Toolbox",
    author="Jin Hyun Cheong, Tiankang Xie, Sophie Byrne, Eshin Jolly, Luke Chang",
    author_email="jcheong0428@gmail.com, eshin.jolly@gmail.com, luke.j.chang@dartmouth.edu",
    url="https://github.com/cosanlab/py-feat",
    packages=find_packages(),
    package_data={"feat": ["resources/*", "tests/*", "tests/data/*"]},
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords=["feat", "face", "facial expression", "emotion"],
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    test_suite="feat/tests",
    **extra_setuptools_args
)
