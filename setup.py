from setuptools import setup, find_packages
from feat.pretrained import get_pretrained_models

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

version = {}
with open("feat/version.py") as f:
    exec(f.read(), version)

extra_setuptools_args = dict(tests_require=["pytest"])


def download_default_models():
    face, landmark, au, emotion, facepose, identity = get_pretrained_models(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
        facepose_model="img2pose",
        identity_model="facenet",
    )


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
    extras_require={
        "default_models": [
            download_default_models,
        ],
    },
    test_suite="feat/tests",
    **extra_setuptools_args
)
