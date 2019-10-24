import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator
from typing import Tuple, List


def train_test_val_split(df: pd.DataFrame,
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


def remove_outliers(df: pd.DataFrame, zscore: int = 3) -> pd.DataFrame:
    """
    Removes all rows from the given DataFrame containing outliers in any of the columns.

    :param df: Input DataFrame.
    :param zscore: z-score to use when calculating outliers.
    :return: The DataFrame with all outliers removed.
    """

    scores = (df - df.mean()) / df.std(ddof=0).values
    return df[(np.abs(scores) < zscore).all(axis=1)]


def create_pipeline(df: pd.DataFrame,
                    imputer_strategy: str = 'mean',
                    imputer_cols: List[str] = None,
                    scaler: BaseEstimator = StandardScaler,
                    scaler_args: dict = None,
                    cat_cols: List[str] = None,
                    copy: bool = True):
    """
    Creates a pipeline performing the following steps:
    - value imputation
    - value scaling
    - one-hot-encoding of categorical values.

    :param df: Input DataFrame.
    :param imputer_strategy: Imputer strategy applied to missing values.
                             Allowed values are ['mean', 'median', 'most_frequent', 'constant'].
    :param imputer_cols: Columns to impute. If no columns are specified all columns will be imputed.
    :param scaler: Scikit-learn scaler to be applied to all values.
    :param scaler_args: Additional arguments forwarded to the specified scaler.
    :param cat_cols: Categorical columns to be one-hot-encoded.
    :param copy: If True, a copy of the input will be created.
    :return: A tuple containing the pipeline and a function returning the columns names after the pipeline has been
             fitted.
    """

    def create_get_feature_names(p, imp, scl, cat):
        def get_feature_names():
            if not hasattr(p, 'transformers_'):
                raise AssertionError('Pipeline is not yet fitted.')

            try:
                cat_names = p.transformers_[2][1].get_feature_names(cat)
            except NotFittedError:
                cat_names = []
            return np.append(imp, np.append(scl, cat_names))

        return get_feature_names

    if scaler_args is None:
        scaler_args = {}

    cat_features = cat_cols if cat_cols else []
    num_features = [c for c in df.select_dtypes(include=[np.number]).columns.values if c not in cat_features]
    imp_features = []

    if imputer_strategy is not None:
        imp_features = imputer_cols if imputer_cols else num_features

    scale_features = [f for f in num_features if f not in imp_features]

    imp_pipeline = Pipeline([
        ('imputer', SimpleImputer(missing_values=np.nan, strategy=imputer_strategy, copy=copy)),
        ('imp_scaler', scaler(**scaler_args))
    ])

    pipeline = ColumnTransformer([
        ('imp', imp_pipeline, imp_features),
        ('scl', scaler(**scaler_args), scale_features),
        ('one_hot', OneHotEncoder(categories='auto'), cat_features)
    ])

    return pipeline, create_get_feature_names(pipeline, imp_features, scale_features, cat_features)
