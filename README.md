# flying-fox-food-shortage

**Manuscript:**

[https://doi.org/10.48550/arXiv.2301.10351](https://doi.org/10.1101/2023.12.01.569640)

**Description**

Food availability determines where animals use space across a landscape and therefore affects the risk of encounters that lead to zoonotic spillover. This relationship is evident in Australian flying foxes (Pteropus spp; fruit bats), where acute food shortages precede clusters of Hendra virus spillovers. Using explainable artificial intelligence, we predicted months of food shortages from climatological and ecological covariates (1996-2022) in eastern Australia. Overall accuracy in predicting months of low food availability on a test set from 2018 up to 2022 reached 93.33% and 92.59% based on climatological and bat-level features, respectively. Seasonality and Oceanic El Niño Index were the most important environmental features, while the number of bats in rescue centers and their body weights were the most important bat-level features. These models support predictive signals up to nine months in advance, facilitating action to mitigate spillover risk.

**Directories:**

    flying-fox-food-shortage
    ├── config
    │   ├── bat_features.txt
    │   ├── env_features.txt
    │   └── rename.txt
    ├── figures
    │   └── *.pdf
    ├── scripts
    │   ├── HyperparameterTuning.py
    │   ├── Metrics.py
    │   └── Plots.py
    ├── FlyingFoxFoodShortage.ipynb
    └── requirements.txt

**Config:**

The `config` folder includes names of bat-level and environmental input features for the GBDT model as well as a renaming dictionary for plotting purposes.

**Figures:**

The `figures` folder includes all modeling and explainable-AI figures used in the manuscript. See `FlyingFoxFoodShortage.ipynb` to generate these figures.

**Scripts:**

The `scripts` folder includes utility scripts implemented in Python that assist in model training, inference, and evaluation. 

- `HyperparameterTuning.py` conducts a grid search over hyperparameter combinations and cross-validation folds to select the optimal set of hyperparameters.
- `Metrics.py` computes performance metrics like accuracy and confusion matrices for model evaluation.
- `Plots.py` generates and saves the relevant modeling and explainable-AI figures used in the manuscript.

**Notebooks:**

The `FlyingFoxFoodShortage.ipynb` notebook is used for data loading and pre-processing, model training, model inference, model evaluation, and figure generation.

**Citation:**

    @article {
          Lagergren2023.12.01.569640,
	      author = {John Lagergren and Manuel Ruiz-Aravena and Daniel J. Becker and Wyatt Madden and Lib Ruytenberg and Andrew Hoegh and Barbara Han and Alison J. Peel and Peggy Eby and Daniel Jacobson and Raina K. Plowright},
	      title = {Environmental and ecological signals predict periods of nutritional stress for Eastern Australian flying fox populations},
	      elocation-id = {2023.12.01.569640},
	      year = {2023},
	      doi = {10.1101/2023.12.01.569640},
	      publisher = {Cold Spring Harbor Laboratory},
	      URL = {https://www.biorxiv.org/content/early/2023/12/04/2023.12.01.569640},
	      eprint = {https://www.biorxiv.org/content/early/2023/12/04/2023.12.01.569640.full.pdf},
	      journal = {bioRxiv}
    }
