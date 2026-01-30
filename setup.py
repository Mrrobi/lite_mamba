from pathlib import Path

import re

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
INIT_PY = ROOT / "lite_mamba" / "__init__.py"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def read_version() -> str:
    text = INIT_PY.read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"\s*$', text, re.M)
    if not match:
        raise RuntimeError("Unable to find __version__ in lite_mamba/__init__.py")
    return match.group(1)

setup(
    name="lite-mamba",
    version=read_version(),
    description="Pure-PyTorch lightweight Mamba with multi-dilated causal conv front-end",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Md Robiuddin",
    author_email="mrrobi040@gmail.com",
    url="https://github.com/Mrrobi/lite_mamba",
    project_urls={
        "Homepage": "https://github.com/Mrrobi/lite_mamba",
    },
    packages=find_packages(exclude=("tests", "docs")),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "einops>=0.6",
    ],
    license="Apache-2.0",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
