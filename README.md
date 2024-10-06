# Interpreting Microbiome Relative Abundance Data Using Symbolic Regression

Code for the preprint ["Interpreting Microbiome Relative Abundance Data Using Symbolic Regression"](add link later).

## Initial setup
Install all dependencies with `pip install -e .`. We recommend creating a separate conda environment (with Python >=3.10) for the installation.

## Repository structure
This repository has the following structure. The main library (`srmb`) is essentially 3 `.py` files that implements the special functions to be used in the function set
of the `SymbolicClassifier` estimator from `gplearn` library, custom fitness functions (explicitly says to optimize the expression for accuracy or F1 or any other
classification performance metric), and some utilities to visualize the learned symbolic tree expression using `networkx` graphs.

For the purposes of reproducing the results in the paper, we also supply all the SR and SRf estimators (for the 20 random runs) as `pickle` objects in the `results_srmb/` directory.
```
.
├── LICENSE
├── README.md
├── data
│   └── data_diet_filtered.csv
├── notebooks
│   ├── data_visualization.ipynb
│   ├── experiment01_baseline_models_accuracy.ipynb
|   ├── experiment02_xgboost_surrogate.ipynb
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

- Experiment 1: #todo
- Experiment 2: #todo
- Visualization of symbolic regression expression tree: #todo

## Minimal use

```python
from srmb.fitness_functions import customacc
from srmb.special_functions import (
    presence, absence, add3, add10, ifelse, ifelseless,
    presence2, absence2,
    presence3, absence
)
from gplearn.genetic import SymbolicClassifier


# SR with special functions
special_functions = [presence, absence, presence2, absence2, ifelse]#, add3, add10]
function_set = ['add', 'sub', 'mul', 'div', 'neg', 'max', 'min'] + special_functions

# ... prepare data X and y
est = SymbolicClassifier(population_size=6000,
                         generations=20,
                         tournament_size=25,

                         init_depth=(2, 6),
                         const_range=(0., 100.),
                         # init_method="full",
                         parsimony_coefficient=0.001,
                         function_set=function_set,

                         stopping_criteria=1.0, metric=customacc, #use custom acc as fitness
                         
                         feature_names=X1.columns.to_list(),
                         # verbose=True,
                         random_state=42)

est.fit(X_train, y_train)

# Visualize the learned expression tree
from IPython.display import display, Image
from srmb.utils import load_sr_models, create_graph, graph_to_jpg

# it will save the image as jpg file and also display if running in a notebook cell
G = create_graph(est)
display(Image(graph_to_jpg(G, path="../results_srmb/viz.jpg"), width=500, unconfined=True))
```

## TODO


