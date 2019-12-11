"""
Utilities for machine learning model selection.
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve


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


def best_precision_for_target_recall(y_true, y_pred_score, target_recall):
    """
    Determines the decision boundary for the best precision given a specified target recall by using
    the precision-recall curve.

    :param y_true: True labels.
    :param y_pred_score: Predicted labels.
    :param target_recall: Target recall.
    :return: Decision boundary.
    """
    _, recalls, thresholds = precision_recall_curve(y_true, y_pred_score)
    return thresholds[np.argmin(recalls >= target_recall)]
