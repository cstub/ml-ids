"""
Utility functions for model selection of Keras models.
"""
import gc
from typing import Tuple
import numpy as np
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras


def cross_val_train(fit_fn,
                    X: np.ndarray,
                    y: np.ndarray,
                    target_transform_fn=id,
                    target_stratify_fn=id,
                    n_splits: int = 3,
                    fit_args: dict = None,
                    random_state: int = None) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Performs stratified cross-validation for a Keras model using the provided fit function.

    :param fit_fn: The function used to fit a model with a given split of the train and test set. Must return a fitted
                   Keras model with its history.
    :param X: Predictor variables.
    :param y: Labels.
    :param target_transform_fn: Function to transform the target labels (e.g. one-hot encoding).
    :param target_stratify_fn: Function to extract the target label to stratify by.
    :param n_splits: Number of cross-validation splits.
    :param fit_args: Arguments to pass to the fit function.
    :param random_state: Random state.
    :return: A triple containing the cross-validation predictions, the true values and a list of history-objects.
    """
    if fit_args is None:
        fit_args = {}

    kfold = StratifiedKFold(n_splits=n_splits, random_state=random_state)

    cv_predictions = None
    cv_y_true = None
    hists = []
    fold = 1

    for train_index, val_index in kfold.split(X, target_stratify_fn(y)):
        print('\nFold {}/{}:'.format(fold, n_splits))
        print('==========')

        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        y_train_ = target_transform_fn(y_train)
        y_val_ = target_transform_fn(y_val)

        keras.backend.clear_session()
        gc.collect()

        model, hist = fit_fn(X_train, y_train_, X_val, y_val_, fit_args, (fold == 1))

        if isinstance(hist, list):
            hists.extend(hist)
        else:
            hists.append(hist)

        if cv_predictions is not None:
            cv_predictions = np.append(cv_predictions, model.predict(X_val), axis=0)
        else:
            cv_predictions = model.predict(X_val)

        if cv_y_true is not None:
            cv_y_true = np.append(cv_y_true, y_val, axis=0)
        else:
            cv_y_true = y_val

        fold = fold + 1

    return cv_predictions, cv_y_true, hists
