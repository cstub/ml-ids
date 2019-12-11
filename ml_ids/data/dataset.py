"""
Utilities to manipulate the CIC-IDS-2018 dataset.
"""
from typing import List
import os
import glob
import numpy as np
import pandas as pd
import ml_ids.data.metadata as md


def remove_inf_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replaces values of type `np.inf` and `-np.inf` in a DataFrame with `null` values.

    :param df: Input DataFrame.
    :return: The DataFrame without `np.inf` and `-np.inf` values.
    """
    inf_columns = [c for c in df.columns if df[df[c] == np.inf][c].count() > 0]
    for col in inf_columns:
        df[col].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def remove_negative_values(df: pd.DataFrame, ignore_cols: List[str] = None) -> pd.DataFrame:
    """
    Removes negative values in a DataFrame with `null` values.

    :param df: Input DataFrame.
    :param ignore_cols: Columns to ignore. Negative values in this columns will be preserved.
    :return: The DataFrame without negative values.
    """
    if ignore_cols is None:
        ignore_cols = []

    numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(ignore_cols).values

    columns = [c for c in numeric_cols if df[df[c] < 0][c].count() > 0]
    for col in columns:
        mask = df[col] < 0
        df.loc[mask, col] = np.nan
    return df


def add_label_category_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column `label_cat` to the DataFrame specifying the category of the label.

    :param df: Input DataFrame.
    :return: The DataFrame containing a new column `label_cat`.
    """
    df[md.COLUMN_LABEL_CAT] = df.label.apply(lambda l: md.LABEL_CAT_MAPPING[l])
    return df


def add_label_is_attack_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column `label_is_attack` to the DataFrame containing a binary indicator specifying if a row is of category
    `benign = 0` or `attack = 1`.

    :param df: Input DataFrame.
    :return: The DataFrame containing a new column `label_is_attack`.
    """
    df[md.COLUMN_LABEL_IS_ATTACK] = df.label.apply(lambda l: 0 if l == md.LABEL_BENIGN else 1)
    return df


def load_dataset_generic(load_df_fn,
                         dataset_path: str,
                         use_cols: List[str] = None,
                         omit_cols: List[str] = None,
                         preserve_neg_value_cols: list = None,
                         transform_data: bool = True) -> pd.DataFrame:
    """
    Loads the dataset from the given path using the supplied function.
    All invalid values (`np.inf`, `-np.inf`, negative) are removed and replaced with `null` for easy imputation.
    Negative values of columns specified in `preserve_neg_value_cols` will be preserved.

    :param load_df_fn: Function used to load the dataset.
    :param dataset_path: Path of the base directory containing all files of the dataset.
    :param use_cols: Columns to load.
    :param omit_cols: Columns to omit.
    :param nrows: Number of rows to load per file.
    :param transform_data: Indicates if data should be manipulated (removal of invalid and negative values).
    :param preserve_neg_value_cols: Columns in which negative values are preserved.
    :return: The dataset as a DataFrame.
    """
    cols = None
    if use_cols:
        cols = use_cols
    if omit_cols:
        cols = [c for c in md.COLUMN_DTYPES.keys() if c not in omit_cols]

    df = load_df_fn(dataset_path, cols)

    if transform_data:
        df = remove_inf_values(df)
        df = remove_negative_values(df, preserve_neg_value_cols)

    if md.COLUMN_LABEL in df.columns:
        df = add_label_category_column(df)
        df = add_label_is_attack_columns(df)

    return df


def load_dataset(dataset_path: str,
                 use_cols: List[str] = None,
                 omit_cols: List[str] = None,
                 nrows: int = None,
                 transform_data: bool = True,
                 preserve_neg_value_cols: list = None) -> pd.DataFrame:
    """
    Loads the dataset in CSV format from the given path.
    All invalid values (`np.inf`, `-np.inf`, negative) are removed and replaced with `null` for easy imputation.
    Negative values of columns specified in `preserve_neg_value_cols` will be preserved.

    :param dataset_path: Path of the base directory containing all files of the dataset.
    :param use_cols: Columns to load.
    :param omit_cols: Columns to omit.
    :param nrows: Number of rows to load per file.
    :param transform_data: Indicates if data should be manipulated (removal of invalid and negative values).
    :param preserve_neg_value_cols: Columns in which negative values are preserved.
    :return: The dataset as a DataFrame.
    """

    def load_csv(path, cols):
        files = glob.glob(os.path.join(path, '*.csv'))
        return pd.concat([pd.read_csv(f, dtype=md.COLUMN_DTYPES, usecols=cols, nrows=nrows) for f in files])

    return load_dataset_generic(load_df_fn=load_csv,
                                dataset_path=dataset_path,
                                use_cols=use_cols,
                                omit_cols=omit_cols,
                                preserve_neg_value_cols=preserve_neg_value_cols,
                                transform_data=transform_data)


def load_dataset_hdf(dataset_path: str,
                     use_cols: List[str] = None,
                     omit_cols: List[str] = None,
                     preserve_neg_value_cols: list = None,
                     transform_data: bool = True,
                     key: str = None) -> pd.DataFrame:
    """
    Loads the dataset stored as a HDF file from the given path.
    All invalid values (`np.inf`, `-np.inf`, negative) are removed and replaced with `null` for easy imputation.
    Negative values of columns specified in `preserve_neg_value_cols` will be preserved.

    :param dataset_path: Path of the base directory containing all files of the dataset.
    :param use_cols: Columns to load.
    :param omit_cols: Columns to omit.
    :param preserve_neg_value_cols: Columns in which negative values are preserved.
    :param transform_data: Indicates if data should be manipulated (removal of invalid and negative values).
    :param key: Group identifier in the HDF store.
    :return: The dataset as a DataFrame.
    """

    def load_hdf(path, cols):
        return pd.read_hdf(path, key=key, columns=cols)

    return load_dataset_generic(load_df_fn=load_hdf,
                                dataset_path=dataset_path,
                                use_cols=use_cols,
                                omit_cols=omit_cols,
                                preserve_neg_value_cols=preserve_neg_value_cols,
                                transform_data=transform_data)
