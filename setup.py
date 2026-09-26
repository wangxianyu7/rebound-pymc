#!/usr/bin/env python

# Inspired by:
# https://hynek.me/articles/sharing-your-labor-of-love-pypi-quick-and-dirty/

import codecs
import os
import re

from setuptools import find_packages, setup

# PROJECT SPECIFIC

NAME = "rebound-pymc"
PACKAGES = find_packages(where="src")
META_PATH = os.path.join("src", "rebound_pymc", "__init__.py")
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3",
]
INSTALL_REQUIRES = ["exoplanet>=0.6.0", "exoplanet-core>=0.2", "rebound>=4.0,<5", "pytensor>=2.20", "numpy<2"]

# END PROJECT SPECIFIC


HERE = os.path.dirname(os.path.realpath(__file__))


def read(*parts):
    with codecs.open(os.path.join(HERE, *parts), "rb", "utf-8") as f:
        return f.read()


def find_meta(meta, meta_file=read(META_PATH)):
    meta_match = re.search(
        r"^__{meta}__ = ['\"]([^'\"]*)['\"]".format(meta=meta), meta_file, re.M
    )
    if meta_match:
        return meta_match.group(1)
    raise RuntimeError("Unable to find __{meta}__ string.".format(meta=meta))


if __name__ == "__main__":
    setup(
        name=NAME,
        version=find_meta("version", meta_file=read(os.path.join("src","rebound_pymc","rebound_pymc_version.py"))),
        author=find_meta("author"),
        author_email=find_meta("email"),
        maintainer=find_meta("author"),
        maintainer_email=find_meta("email"),
        url=find_meta("uri"),
        license=find_meta("license"),
        description=find_meta("description"),
        long_description=read("README.md"),
        long_description_content_type="text/markdown",
        packages=PACKAGES,
        package_dir={"": "src"},
        include_package_data=True,
        package_data={"rebound_pymc": ["*.cc", "*.h", "*.py"]},
        install_requires=INSTALL_REQUIRES,
        classifiers=CLASSIFIERS,
        zip_safe=False,
    )
