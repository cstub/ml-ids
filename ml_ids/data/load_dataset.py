import numpy as np
import pandas as pd
import glob
import os
from typing import List

COLUMN_DTYPES = types = {
    'dst_port': 'uint32',
    'protocol': 'uint8',
    'timestamp': 'object',
    'flow_duration': 'int64',
    'tot_fwd_pkts': 'uint32',
    'tot_bwd_pkts': 'uint32',
    'totlen_fwd_pkts': 'uint32',
    'totlen_bwd_pkts': 'uint32',
    'fwd_pkt_len_max': 'uint16',
    'fwd_pkt_len_min': 'uint16',
    'fwd_pkt_len_mean': 'float32',
    'fwd_pkt_len_std': 'float32',
    'bwd_pkt_len_max': 'uint16',
    'bwd_pkt_len_min': 'uint16',
    'bwd_pkt_len_mean': 'float32',
    'bwd_pkt_len_std': 'float32',
    'flow_byts_s': 'float64',
    'flow_pkts_s': 'float64',
    'flow_iat_mean': 'float32',
    'flow_iat_std': 'float32',
    'flow_iat_max': 'int64',
    'flow_iat_min': 'int64',
    'fwd_iat_tot': 'int64',
    'fwd_iat_mean': 'float32',
    'fwd_iat_std': 'float32',
    'fwd_iat_max': 'int64',
    'fwd_iat_min': 'int64',
    'bwd_iat_tot': 'uint32',
    'bwd_iat_mean': 'float32',
    'bwd_iat_std': 'float32',
    'bwd_iat_max': 'uint32',
    'bwd_iat_min': 'uint32',
    'fwd_psh_flags': 'uint8',
    'bwd_psh_flags': 'uint8',
    'fwd_urg_flags': 'uint8',
    'bwd_urg_flags': 'uint8',
    'fwd_header_len': 'uint32',
    'bwd_header_len': 'uint32',
    'fwd_pkts_s': 'float32',
    'bwd_pkts_s': 'float32',
    'pkt_len_min': 'uint16',
    'pkt_len_max': 'uint16',
    'pkt_len_mean': 'float32',
    'pkt_len_std': 'float32',
    'pkt_len_var': 'float32',
    'fin_flag_cnt': 'uint8',
    'syn_flag_cnt': 'uint8',
    'rst_flag_cnt': 'uint8',
    'psh_flag_cnt': 'uint8',
    'ack_flag_cnt': 'uint8',
    'urg_flag_cnt': 'uint8',
    'cwe_flag_count': 'uint8',
    'ece_flag_cnt': 'uint8',
    'down_up_ratio': 'uint16',
    'pkt_size_avg': 'float32',
    'fwd_seg_size_avg': 'float32',
    'bwd_seg_size_avg': 'float32',
    'fwd_byts_b_avg': 'uint8',
    'fwd_pkts_b_avg': 'uint8',
    'fwd_blk_rate_avg': 'uint8',
    'bwd_byts_b_avg': 'uint8',
    'bwd_pkts_b_avg': 'uint8',
    'bwd_blk_rate_avg': 'uint8',
    'subflow_fwd_pkts': 'uint32',
    'subflow_fwd_byts': 'uint32',
    'subflow_bwd_pkts': 'uint32',
    'subflow_bwd_byts': 'uint32',
    'init_fwd_win_byts': 'int32',
    'init_bwd_win_byts': 'int32',
    'fwd_act_data_pkts': 'uint32',
    'fwd_seg_size_min': 'uint8',
    'active_mean': 'float32',
    'active_std': 'float32',
    'active_max': 'uint32',
    'active_min': 'uint32',
    'idle_mean': 'float32',
    'idle_std': 'float32',
    'idle_max': 'uint64',
    'idle_min': 'uint64',
    'label': 'category'
}

LABEL_BENIGN = 'Benign'
LABEL_CAT_MAPPING = {
    'Benign': 0,
    'Bot': 1,
    'Brute Force -Web': 2,
    'Brute Force -XSS': 3,
    'DoS attacks-GoldenEye': 4,
    'DoS attacks-Hulk': 5,
    'DoS attacks-SlowHTTPTest': 6,
    'DoS attacks-Slowloris': 7,
    'DDOS attack-HOIC': 8,
    'DDOS attack-LOIC-UDP': 9,
    'DDoS attacks-LOIC-HTTP': 10,
    'FTP-BruteForce': 11,
    'Infilteration': 12,
    'SQL Injection': 13,
    'SSH-Bruteforce': 14,
    'DDOS LOIT': 15,
    'Heartbleed': 16,
    'PortScan': 17
}


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

    df['label_cat'] = df.label.apply(lambda l: LABEL_CAT_MAPPING[l])
    return df


def add_label_is_attack_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds the column `label_is_attack` to the DataFrame containing a binary indicator specifying if a row is of category
    `benign = 0` or `attack = 1`.

    :param df: Input DataFrame.
    :return: The DataFrame containing a new column `label_is_attack`.
    """

    df['label_is_attack'] = df.label.apply(lambda l: 0 if l == LABEL_BENIGN else 1)
    return df


def load_dataset(dataset_path: str,
                 use_cols: List[str] = None,
                 omit_cols: List[str] = None,
                 nrows: int = None,
                 preserve_neg_value_cols: list = None) -> pd.DataFrame:
    """
    Loads the dataset from the given path.
    All invalid values (`np.inf`, `-np.inf`, negative) are removed and replaced with `null` for easy imputation.
    Negative values of columns specified in `preserve_neg_value_cols` will be preserved.

    :param dataset_path: Path of the base directory containing all files of the dataset.
    :param use_cols: Columns to load.
    :param omit_cols: Columns to omit.
    :param nrows: Number of rows to load per file.
    :param preserve_neg_value_cols: Columns in which negative values are preserved.
    :return: The dataset as a DataFrame.
    """

    cols = None
    if use_cols:
        cols = use_cols
    if omit_cols:
        cols = [c for c in COLUMN_DTYPES.keys() if c not in omit_cols]

    files = glob.glob(os.path.join(dataset_path, '*.csv'))

    df = pd.concat([pd.read_csv(f, dtype=COLUMN_DTYPES, usecols=cols, nrows=nrows) for f in files])

    df = remove_inf_values(df)
    df = remove_negative_values(df, preserve_neg_value_cols)

    if 'label' in df.columns:
        df = add_label_category_column(df)
        df = add_label_is_attack_columns(df)

    return df
