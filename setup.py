"""
Setup script for the CAT‑Net package.

This allows you to install the core CAT‑Net Python package with
``pip install -e .``.  It pulls dependencies from ``requirements.txt``
to ensure the package has the same requirements as the scripts and
examples included in this repository.
"""

from pathlib import Path
from setuptools import setup, find_packages


def parse_requirements(filename: str) -> list[str]:
    """Parse a requirements file into a list of strings.

    Comments and empty lines are ignored.  Lines starting with
    ``-r`` or ``--requirement`` are not supported here since this
    project uses a single requirements file.
    """
    reqs: list[str] = []
    for line in Path(filename).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


install_requires = parse_requirements("requirements.txt")


setup(
    name="catnet",
    version="1.0.0",
    description=(
        "CAT‑Net: Channel and Self‑Attention TCN for Frame‑Level Overlapping "
        "Speech Detection"
    ),
    author="Yassin Terraf and Youssef Iraqi",
    # Package only the top‑level ``models`` directory.  The repository no longer
    # contains a nested ``catnet`` package; models are located directly under
    # ``catnet/models``.
    packages=find_packages(include=["models", "models.*"]),
    python_requires=">=3.8",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)