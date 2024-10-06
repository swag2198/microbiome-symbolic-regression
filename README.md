# Interpreting Microbiome Relative Abundance Data Using Symbolic Regression

Code for the preprint ["Interpreting Microbiome Relative Abundance Data Using Symbolic Regression"](add link later).

## Initial setup
Install all dependencies with `pip install -e .`. We recommend creating a separate conda environment (with Python >=3.10) for the installation.

## Repository structure
This repository has the following structure. The main library (`srmb`) is essentially 3 `.py` files that implements the special functions to be used in the function set
of the `SymbolicClassifier` estimator from `gplearn` library, custom fitness functions (explicitly says to optimize the expression for accuracy or F1 or any other
classification performance metric), and some utilities to visualize the learned symbolic tree expression using `networkx` graphs.
```
.
├── LICENSE
├── README.md
├── data
│   └── data_diet_filtered.csv
├── notebooks
│   ├── data_visualization.ipynb
│   ├── experiment01_baseline_models_accuracy.ipynb
    ├── experiment02_xgboost_surrogate.ipynb
│   └── sr_model_graph_visualization.ipynb
├── results_srmb
│   ├── sr_special_models/
│   └── sr_vanilla_models/
├── setup.py
└── srmb
   ├── fitness_functions.py
   ├── special_functions.py
   └── utils.py
```

## Reproducing results of experiments

## Minimal use

## TODO


