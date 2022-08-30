from os import path
from pathlib import Path

from setuptools import find_packages, setup

this_dir = path.abspath(path.dirname(__file__))

with open(path.join(this_dir, "README.md"), encoding="utf-8") as f:
    long_description = f.read()


def match_prefix(lib, match_list):
    if not match_list:
        return False
    for m in match_list:
        if lib.strip().startswith(m):
            return True
    return False


def reqs(file, without=None):
    with open(Path(__file__).resolve().parent / file) as rf:
        return [
            r.strip()
            for r in rf.readlines()
            if not (r.startswith("#") or r.startswith("\n"))
            and not match_prefix(r, without)
        ]


compat_reqs = ['dataclasses==0.7;python_version<"3.7"']

utils_reqs = reqs("utils-requirements.txt")
test_reqs = reqs("test-requirements.txt")

setup(
    name="gretel-synthetics",
    author="Gretel Labs, Inc.",
    author_email="open-source@gretel.ai",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Synthetic Data Generation with optional Differential Privacy",
    url="https://github.com/gretelai/gretel-synthetics",
    license="https://gretel.ai/license/source-available-license",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.7",
    install_requires=(
        reqs("requirements.txt", without=["tensorflow==", "torch=="]) + compat_reqs
    ),
    extras_require={"all": utils_reqs, "utils": utils_reqs, "test": test_reqs},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free To Use But Restricted",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
