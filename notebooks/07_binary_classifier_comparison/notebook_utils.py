import numpy as np
import gc
from ml_ids.model_selection import split_x_y, train_val_test_split
from ml_ids.transform.sampling import upsample_minority_classes, downsample
from ml_ids.transform.preprocessing import create_pipeline
from collections import Counter


def get_best_model_path(trials, model_path_var='model_path'):
    return trials.results[np.argmin(trials.losses())][model_path_var]


def print_trial_results(trials, best_run, model_path_var='model_path'):
    best_model_path = get_best_model_path(trials, model_path_var)

    print('Best validation score: {}'.format(-np.min(trials.losses())))
    print('Best model path: {}\n'.format(best_model_path))
    print('Best model parameters:')
    print('======================')
    print(best_run)


def transform_data(dataset,
                   attack_samples,
                   imputer_strategy,
                   scaler,
                   benign_samples=None,
                   random_state=None):

    cols_to_impute = dataset.columns[dataset.isna().any()].tolist()

    train_data, val_data, test_data = train_val_test_split(dataset,
                                                           val_size=0.1,
                                                           test_size=0.1,
                                                           stratify_col='label_cat',
                                                           random_state=random_state)

    if benign_samples:
        train_data = downsample(train_data, default_nr_samples=benign_samples, random_state=random_state)

    X_train_raw, y_train = split_x_y(train_data)
    X_val_raw, y_val = split_x_y(val_data)
    X_test_raw, y_test = split_x_y(test_data)

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train_raw.shape))
    print('Val:      {}'.format(X_val_raw.shape))
    print('Test:     {}'.format(X_test_raw.shape))

    print('\nTraining labels:')
    print('================')
    print(y_train.label.value_counts())
    print('\nValidation labels:')
    print('==================')
    print(y_val.label.value_counts())
    print('\nTest labels:')
    print('============')
    print(y_test.label.value_counts())

    del train_data, val_data, test_data
    gc.collect()

    pipeline, get_col_names = create_pipeline(X_train_raw,
                                              imputer_strategy=imputer_strategy,
                                              imputer_cols=cols_to_impute,
                                              scaler=scaler)

    X_train = pipeline.fit_transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    column_names = get_col_names()

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train.shape))
    print('Val:      {}'.format(X_val.shape))
    print('Test:     {}'.format(X_test.shape))

    print('\nMissing values:')
    print('===============')
    print('Training: {}'.format(np.count_nonzero(np.isnan(X_train))))
    print('Val:      {}'.format(np.count_nonzero(np.isnan(X_val))))
    print('Test:     {}'.format(np.count_nonzero(np.isnan(X_test))))

    print('\nScaling:')
    print('========')
    print('Training: min={}, max={}'.format(np.min(X_train), np.max(X_train)))
    print('Val:      min={}, max={}'.format(np.min(X_val), np.max(X_val)))
    print('Test:     min={}, max={}'.format(np.min(X_test), np.max(X_test)))

    X_train, y_train = upsample_minority_classes(X_train,
                                                 y_train,
                                                 min_samples=attack_samples,
                                                 random_state=random_state)

    print('Samples:')
    print('========')
    print('Training: {}'.format(X_train.shape))

    print('\nTraining labels:')
    print('================')
    print(Counter(y_train))

    return X_train, y_train, X_val, y_val, X_test, y_test, column_names
