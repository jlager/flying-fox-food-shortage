import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Tuple
from sklearn.preprocessing import StandardScaler

def hyperparameter_tuning(
    model_class,
    model_hyper: dict,
    model_fixed: dict,
    train_hyper: dict,
    train_fixed: dict,
    x: np.ndarray,
    y: np.ndarray,
    train_indices: list,
    val_indices: list,
    weights: list = None,
    verbose: bool = True,
    normalize: bool = False,
) -> Tuple[dict, dict, list, list]:

    '''
    This function performs hyperparameter tuning for a given model class.

    model_class must have the following methods:
        - fit(X, y, eval_set=(X_val, y_val), **kwargs) -> None
        - evaluate(X, y) -> float

    For example, the model class could be a wrapper for a LightGBM model, 
    where the fit method is already implemented, and the evaluate method
    can be implemented as follows:

        def evaluate(X, y):
            score = min(self.evals_result_['valid_0']['binary_logloss'])

    Note that this example applies to a LightGBM classifier, and that the
    evaluate method returns the minimum validation log loss, which was 
    already computed during training.

    Parameters
    ----------
    model_class : class
        Class of model to use for training and evaluation, must include
        fit and evaluate methods, see above.
    model_hyper : dict
        Dictionary of tunable hyperparameters for the model. Each value
        must be a list of hyperparameter values to try.
    model_fixed : dict
        Dictionary of fixed hyperparameters for the model. Each value
        must be a single hyperparameter value.
    train_hyper : dict
        Dictionary of tunable hyperparameters for the training procedure.
        Each value must be a list of hyperparameter values to try.
    train_fixed : dict
        Dictionary of fixed hyperparameters for the training procedure.
        Each value must be a single hyperparameter value.
    x : np.ndarray
        Input variables for training and evaluation [n_samples, n_features].
    y : np.ndarray
        Target variable for training and evaluation [n_samples].
    train_indices : list
        List of index sets to use for training. Each index set must be
        an iterable of indices (i.e., rows of the dataset) that can be
        used to index numpy arrays.
    val_indices : list
        List of index sets to use for validation. Each index set must be
        an iterable of indices (i.e., rows of the dataset) that can be
        used to index numpy arrays.
    weights : list, optional
        List of weights to use for weighted average of scores for each
        hyperparameter combination.
    verbose : bool, optional
        Whether to print results and use status bar during tuning.
    normalize : bool, optional
        Whether to center and scale the input features.

    Returns
    -------
    best_model_params : dict
        Dictionary of best model hyperparameters.
    best_train_params : dict
        Dictionary of best training hyperparameters.
    model_list : list
        List of models for each hyperparameter and train/val combination.
    score_list : list
        List of scores for each hyperparameter and train/val combination.
    '''

    # unpack hyperparameters
    model_h_keys = list(model_hyper.keys())
    model_h_vals = list(model_hyper.values())
    train_h_keys = list(train_hyper.keys())
    train_h_vals = list(train_hyper.values())
    
    # get number of hyperparameter combinations
    model_h_lens = [len(h) for h in model_h_vals]
    train_h_lens = [len(h) for h in train_h_vals]
    n_combinations = np.prod(model_h_lens + train_h_lens)

    # initialize list to store model scores
    model_list, score_list = [], []

    # loop over hyperparameter combinations
    loop = tqdm(range(n_combinations)) if verbose else range(n_combinations)
    for i in loop:

        # unravel hyperparameter values
        indices = np.unravel_index(i, model_h_lens + train_h_lens)
        model_hyper_i = {k: v[i] for k, v, i in zip(
            model_h_keys, 
            model_h_vals, 
            indices[:len(model_h_keys)])}
        train_hyper_i = {k: v[i] for k, v, i in zip(
            train_h_keys, 
            train_h_vals, 
            indices[len(model_h_keys):])}

        # fit models on train/val sets and store scores
        models, scores = [], []
        for t_idx, v_idx in zip(train_indices, val_indices):
            
            # initialize model with hyperparameter combination
            model = model_class(**model_hyper_i, **model_fixed)

            # normalize input features
            if normalize:
                scaler = StandardScaler().fit(x[t_idx])
            else:
                class IdentifyScaler:
                    def transform(self, X):
                        return X
                scaler = IdentifyScaler()

            # fit model on train/val set
            model.fit(
                X=scaler.transform(x[t_idx]),
                y=y[t_idx],
                eval_set=(scaler.transform(x[v_idx]), y[v_idx]),
                **train_hyper_i,
                **train_fixed,
            )

            # evaluate model on val set
            score = model.evaluate(scaler.transform(x[v_idx]), y[v_idx])

            # store model and score for train/val set
            models.append(model)
            scores.append(score)

        # store scores for hyperparameter combination
        model_list.append(models)
        score_list.append(scores)

    # compute weighted scores for each hyperparameter combination
    eval_matrix = np.zeros(model_h_lens + train_h_lens)
    for i, scores in enumerate(score_list):
        indices = np.unravel_index(i, model_h_lens + train_h_lens)
        eval_matrix[indices] = np.average(scores, weights=weights)

    # get best hyperparameters
    argmin = np.nanargmin(eval_matrix)
    indices = np.unravel_index(argmin, eval_matrix.shape)
    best_model_params = {k: v[i] for k, v, i in zip(
        model_h_keys,
        model_h_vals,
        indices[:len(model_h_keys)])}
    best_train_params = {k: v[i] for k, v, i in zip(
        train_h_keys,
        train_h_vals,
        indices[len(model_h_keys):])}
    
    # print results
    if verbose:
        print()
        print('best model parameters:')
        j = max([len(k) for k in model_h_keys]) if len(model_h_keys) > 0 else 0
        for k, v in best_model_params.items():
            print(f'{k.ljust(j)} {v}')
        print()
        print('best train parameters:')
        j = max([len(k) for k in train_h_keys]) if len(train_h_keys) > 0 else 0
        for k, v in best_train_params.items():
            print(f'{k.ljust(j)} {v}')
        print()
        print('eval score:')
        print(f'score: {eval_matrix[indices]}')
        print(f'best index: {argmin}')
        print()

    return best_model_params, best_train_params, model_list, score_list