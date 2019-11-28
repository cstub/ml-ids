import pytest
import pandas as pd
import numpy as np
import os
from ml_ids import conf
from ml_ids.data.dataset import load_dataset


@pytest.fixture
def val_data():
    validation_data_path = os.path.join(conf.TEST_DATA_DIR, 'validation.csv')
    return pd.read_csv(validation_data_path)


def inf_value_count(df):
    return df[(df == np.inf) | (df == -np.inf)].count().sum()


def neg_value_count(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.values
    df_num = df[numeric_cols]
    return df_num[df_num < 0].count().sum()


def nan_value_count(df):
    return df.isna().sum().sum()


def negative_value_columns(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns.values
    return [c for c in numeric_cols if df[df[c] < 0][c].count() > 0]


def test_loaded_dataset_must_not_contain_inf_values():
    df = load_dataset(conf.TEST_DATA_DIR)

    assert inf_value_count(df) == 0


def test_loaded_dataset_must_not_contain_negative_values():
    df = load_dataset(conf.TEST_DATA_DIR)

    assert neg_value_count(df) == 0


def test_loaded_dataset_must_not_contain_negative_values_except_excluded_cols():
    df = load_dataset(conf.TEST_DATA_DIR, preserve_neg_value_cols=['init_fwd_win_byts', 'init_bwd_win_byts'])

    assert neg_value_count(df) != 0
    assert set(negative_value_columns(df)) == {'init_bwd_win_byts', 'init_fwd_win_byts'}


def test_loaded_dataset_must_contain_label_category():
    df = load_dataset(conf.TEST_DATA_DIR)

    assert len(df.label_cat.value_counts()) == len(df.label.value_counts())


def test_loaded_dataset_must_contain_label_is_attack():
    df = load_dataset(conf.TEST_DATA_DIR)

    all_sample_count = len(df)
    benign_sample_count = len(df[df.label == 'Benign'])
    attack_sample_count = all_sample_count - benign_sample_count

    assert len(df[df.label_is_attack == 0]) == benign_sample_count
    assert len(df[df.label_is_attack == 1]) == attack_sample_count


def test_loaded_dataset_must_replace_invalid_value_with_nan(val_data):
    df = load_dataset(conf.TEST_DATA_DIR)

    inf_value_c = inf_value_count(val_data)
    neg_value_c = neg_value_count(val_data)

    assert (inf_value_c + neg_value_c) == nan_value_count(df)


def test_loaded_dataset_must_contain_only_specified_columns():
    df = load_dataset(conf.TEST_DATA_DIR, use_cols=['dst_port'])

    assert df.columns == ['dst_port']


def test_loaded_dataset_must_omit_specified_columns():
    df = load_dataset(conf.TEST_DATA_DIR, omit_cols=['dst_port'])

    assert 'dst_port' not in df.columns
