import pytest
import numpy as np
from numpy.testing import assert_array_equal
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler

from ml_ids import conf
from ml_ids.data.dataset import load_dataset
from ml_ids.model_selection import split_x_y
from ml_ids.transform.preprocessing import create_pipeline


@pytest.fixture
def feature_df():
    df = load_dataset(conf.TEST_DATA_DIR, omit_cols=['timestamp'])
    X, _ = split_x_y(df)
    return X


def nan_value_count(x):
    return np.count_nonzero(np.isnan(x))


def test_pipeline_must_impute_all_missing_values(feature_df):
    pipeline, _ = create_pipeline(feature_df,
                                  imputer_strategy='mean',
                                  scaler=FunctionTransformer,
                                  scaler_args={'validate': False})
    transformed = pipeline.fit_transform(feature_df)

    assert nan_value_count(feature_df.values) != 0
    assert nan_value_count(transformed) == 0


def test_pipeline_must_impute_selected_columns_only(feature_df):
    pipeline, _ = create_pipeline(feature_df,
                                  imputer_strategy='mean',
                                  imputer_cols=['flow_duration', 'flow_pkts_s'],
                                  scaler=FunctionTransformer,
                                  scaler_args={'validate': False})

    missing_vals_selected_columns = \
        nan_value_count(feature_df.flow_duration.values) + nan_value_count(feature_df.flow_pkts_s.values)

    transformed = pipeline.fit_transform(feature_df)

    assert nan_value_count(transformed) == (nan_value_count(feature_df.values) - missing_vals_selected_columns)


def test_pipeline_must_not_impute_values_if_imputer_strategy_none(feature_df):
    pipeline, get_col_names = create_pipeline(feature_df,
                                              imputer_strategy=None,
                                              scaler=FunctionTransformer,
                                              scaler_args={'validate': False})

    transformed = pipeline.fit_transform(feature_df)

    assert nan_value_count(feature_df.values) == nan_value_count(transformed)
    assert len(feature_df.columns) == len(get_col_names())


def test_pipeline_must_reorder_columns(feature_df):
    pipeline, get_col_names = create_pipeline(feature_df,
                                              imputer_strategy='mean',
                                              imputer_cols=['flow_duration', 'flow_pkts_s'],
                                              scaler=FunctionTransformer,
                                              scaler_args={'validate': False})

    _ = pipeline.fit_transform(feature_df)
    column_names = get_col_names()

    assert len(feature_df.columns) == len(column_names)
    assert_array_equal(column_names[:2], ['flow_duration', 'flow_pkts_s'])


def test_pipeline_must_impute_all_missing_values_with_mean(feature_df):
    pipeline, get_col_names = create_pipeline(feature_df,
                                              imputer_strategy='mean',
                                              scaler=FunctionTransformer,
                                              scaler_args={'validate': False})
    transformed = pipeline.fit_transform(feature_df)

    col_idx = np.where(get_col_names() == 'flow_duration')[0]
    nan_idx = np.where(np.isnan(feature_df.flow_duration.values))[0]

    assert len(nan_idx) == 10
    assert np.unique(transformed[nan_idx, col_idx]) == feature_df.flow_duration.mean()


def test_pipeline_must_impute_all_missing_values_with_median(feature_df):
    pipeline, get_col_names = create_pipeline(feature_df,
                                              imputer_strategy='median',
                                              scaler=FunctionTransformer,
                                              scaler_args={'validate': False})
    transformed = pipeline.fit_transform(feature_df)

    col_idx = np.where(get_col_names() == 'flow_duration')[0]
    nan_idx = np.where(np.isnan(feature_df.flow_duration.values))[0]

    assert len(nan_idx) == 10
    assert np.unique(transformed[nan_idx, col_idx]) == feature_df.flow_duration.median()


def test_pipeline_must_scale_all_values(feature_df):
    pipeline, _ = create_pipeline(feature_df, scaler=MinMaxScaler)
    transformed = pipeline.fit_transform(feature_df)

    assert np.min(transformed) == 0
    assert np.max(transformed) == 1


def test_pipeline_must_one_hot_encode_categorical_values(feature_df):
    nr_categories = 3
    pipeline, _ = create_pipeline(feature_df, cat_cols=['protocol'])
    transformed = pipeline.fit_transform(feature_df)

    one_hot_encoded = transformed[:, -nr_categories:]

    print(np.unique(one_hot_encoded))

    assert transformed.shape[1] == feature_df.shape[1] + (nr_categories - 1)
    assert_array_equal(np.unique(one_hot_encoded), [0., 1.])
