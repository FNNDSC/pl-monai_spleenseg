from setuptools import setup
import re

_version_re = re.compile(r"(?<=^__version__ = (\"|'))(.+)(?=\"|')")


def get_version(rel_path: str) -> str:
    """
    Searches for the ``__version__ = `` line in a source code file.

    https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
    """
    with open(rel_path, "r") as f:
        matches = map(_version_re.search, f)
        filtered = filter(lambda m: m is not None, matches)
        version = next(filtered, None)
        if version is None:
            raise RuntimeError(f"Could not find __version__ in {rel_path}")
        return version.group(0)


requirements = []
with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f.readlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def readme():
    with open("README.rst") as f:
        return f.read()


setup(
    name="spleenseg",
    version=get_version("spleenseg/spleenseg.py"),
    description="A ChRIS DS plugin heavily hacked off project MONAI's spleen segmenation notebook",
    long_description=readme(),
    author="FNNDSC",
    author_email="dev@babyMRI.org",
    url="https://github.com/FNNDSC/pl-monai_spleenseg",
    packages=[
        "spleenseg",
        "spleenseg/core",
        "spleenseg/plotting",
        "spleenseg/transforms",
    ],
    install_requires=requirements,
    data_files=[("", ["requirements.txt"])],
    license="MIT",
    entry_points={"console_scripts": ["spleenseg = spleenseg.spleenseg:main"]},
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    extras_require={"none": [], "dev": ["pytest~=7.1"]},
)
