# Interpreting Microbiome Relative Abundance Data Using Symbolic Regression

</div>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#initial-setup">Initial setup</a></li>
    <li><a href="#repository-structure">Repository structure</a></li>
    <li><a href="#reproducing-results-of-experiments">Reproducing results of experiments</a></li>
    <li><a href="#minimal-use">Minimal use</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>

<!-- <li>
      <a href="#description">Description</a>
      <ul>
        <li><a href="#dependencies">Dependencies</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
</li> -->


Code for the preprint ["Interpreting Microbiome Relative Abundance Data Using Symbolic Regression"](add link later).

<p align="center">
  <img src="https://github.com/swag2198/microbiome-symbolic-regression/blob/main/results_srmb/xgboost_distilled_sr.jpg?raw=true" alt="alt text"/>
</p>

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

- Experiment 1: To reproduce the results in Table 1 in the paper, see this notebook [experiment01_baseline_models_accuracy.ipynb](https://github.com/swag2198/microbiome-symbolic-regression/blob/main/notebooks/experiment01_baseline_models_accuracy.ipynb)
- Experiment 2: For the knowledge distillation experiment in Section 4.3 of the paper, see this notebook [experiment02_xgboost_surrogate.ipynb](https://github.com/swag2198/microbiome-symbolic-regression/blob/main/notebooks/experiment02_xgboost_surrogate.ipynb)
- Visualization of symbolic regression expression tree: creating the tree visualization from the learned symbolic regression expression [sr_model_graph_visualization.ipynb](https://github.com/swag2198/microbiome-symbolic-regression/blob/main/notebooks/sr_model_graph_visualization.ipynb)
- Extraction of top bacteria responsible for CRC: #todo

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

## References
The code in this repository is heavily based on the [`gplearn`](https://github.com/trevorstephens/gplearn) repository.

## TODO
