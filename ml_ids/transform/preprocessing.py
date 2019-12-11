"""
Utilities for data pre-processing.
"""
from typing import List
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.exceptions import NotFittedError
from sklearn.base import BaseEstimator


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
    imp_features: List[str] = []

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
