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


doc_reqs = reqs("requirements/docs.txt")
base_reqs = reqs("requirements/base.txt")
utils_reqs = reqs("requirements/utils.txt")
test_reqs = reqs("requirements/test.txt")
torch_reqs = reqs("requirements/torch.txt")
tf_reqs = reqs("requirements/tensorflow.txt")
all_reqs = [base_reqs, utils_reqs, torch_reqs, tf_reqs]

setup(
    name="gretel-synthetics",
    author="Gretel Labs, Inc.",
    author_email="support@gretel.ai",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    description="Synthetic Data Generation with optional Differential Privacy",
    url="https://github.com/gretelai/gretel-synthetics",
    license="https://gretel.ai/license/source-available-license",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.9",
    install_requires=base_reqs,
    extras_require={
        "all": [req for reqs in all_reqs for req in reqs],
        "utils": utils_reqs,
        "test": test_reqs,
        "torch": torch_reqs,
        "tensorflow": tf_reqs,
        "docs": doc_reqs,
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free To Use But Restricted",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
