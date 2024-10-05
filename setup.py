from setuptools import find_packages, setup

NAME = "srmb"
DESCRIPTION = "Interpreting Microbiome Relative Abundance Data Using Symbolic Regression"
URL = "NONE"
AUTHOR = "Swagatam Haldar, Vadim Borisov"
REQUIRES_PYTHON = ">=3.10.0"

REQUIRED = [
    "numpy",
    "torch",
    "matplotlib",
    "SciencePlots",
    "scikit-learn",
    "pandas",
    "scipy",
    "seaborn",
    "networkx",
    "pygraphviz",
    "gplearn",
    "xgboost",
    "pytest"
]

EXTRAS = {
    "dev": [
        "autoflake",
        "black",
        "deepdiff",
        "flake8",
        "isort",
        "ipykernel",
        "jupyter",
        "pep517",
        "pytest",
        "pyyaml",
    ],
}

setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="AGPLv3",
)