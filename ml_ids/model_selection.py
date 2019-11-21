import numpy as np
import pandas as pd
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from tensorflow import keras
from typing import Tuple, List


def train_val_test_split(df: pd.DataFrame,
                         val_size: float = 0.1,
                         test_size: float = 0.1,
                         stratify_col: str = None,
                         random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the given DataFrame into three parts used for:
    - training
    - validation
    - test

    :param df: Input DataFrame.
    :param val_size: Size of validation set.
    :param test_size: Size of test set.
    :param stratify_col: Column to stratify.
    :param random_state: Random state.
    :return: A triple containing (`train`, `val`, `test`) sets.
    """
    assert (val_size + test_size) < 1, 'Sum of validation and test size must not be > 1.'

    df_stratify = df[stratify_col] if stratify_col else None
    df_train, df_hold = train_test_split(df,
                                         test_size=(val_size + test_size),
                                         stratify=df_stratify,
                                         random_state=random_state)

    df_hold_stratify = df_hold[stratify_col] if stratify_col else None
    df_val, df_test = train_test_split(df_hold,
                                       test_size=test_size / (val_size + test_size),
                                       stratify=df_hold_stratify,
                                       random_state=random_state)

    return df_train, df_val, df_test


def split_x_y(df: pd.DataFrame, y_cols: List[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the given DataFrame into a DataFrame `X` containing the predictor variables and a DataFrame 'y' containing
    the labels y.

    :param df: Input DataFrame.
    :param y_cols: Columns to use in the labels DataFrame `y`.
    :return: A tuple containing the DataFrames (`X`, `y`).
    """
    if y_cols is None:
        y_cols = ['label', 'label_cat', 'label_is_attack']
    return df.drop(columns=y_cols), df[y_cols]


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


def best_precision_for_target_recall(y_true, y_pred_score, target_recall):
    """
    Determines the decision boundary for the best precision given a specified target recall by using
    the precision-recall curve.

    :param y_true: True labels.
    :param y_pred_score: Predicted labels.
    :param target_recall: Target recall.
    :return: Decision boundary.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_score)
    return thresholds[np.argmin(recalls >= target_recall)]
