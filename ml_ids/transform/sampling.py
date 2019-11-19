import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC
from typing import Tuple, List


def upsample_minority_classes(X: np.ndarray,
                              y: pd.DataFrame,
                              min_samples: int,
                              random_state: int,
                              cat_cols: List[int] = None,
                              n_jobs: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    Synthetic up-sampling of minority classes using `imblearn.over_sampling.SMOTE`.

    :param X: Predictor variables.
    :param y: Labels.
    :param min_samples: Minimum samples of each class.
    :param random_state: Random state.
    :param cat_cols: Column indices of categorical features.
    :param n_jobs: Number of threads to use.
    :return: A tuple containing the up-sampled X and y values.
    """
    cnts = y.label_cat.value_counts()
    sample_dict = {}

    for i in np.unique(y.label_cat):
        sample_dict[i] = max(cnts[i], min_samples)

    if cat_cols:
        sm = SMOTENC(sampling_strategy=sample_dict,
                     categorical_features=cat_cols,
                     n_jobs=n_jobs,
                     random_state=random_state)
    else:
        sm = SMOTE(sampling_strategy=sample_dict, n_jobs=n_jobs, random_state=random_state)

    X_s, y_s = sm.fit_resample(X, y.label_cat)
    return X_s, y_s


def create_sample_dict(df: pd.DataFrame,
                       default_nr_samples: int,
                       samples_per_label: dict = None) -> dict:
    """
    Creates a dictionary containing the number of samples per label.

    :param df: Input DataFrame.
    :param default_nr_samples: Default number of samples per label.
    :param samples_per_label: Number of samples for specific labels.
    :return: Dictionary containing the number of samples per label.
    """
    if samples_per_label is None:
        samples_per_label = {}

    sample_dict = df.label_cat.value_counts().to_dict()

    for l in sample_dict.keys():
        requested_samples = samples_per_label[l] if l in samples_per_label else default_nr_samples
        existing_samples = sample_dict[l] if l in sample_dict else 0
        sample_dict[l] = min(requested_samples, existing_samples)

    return sample_dict


def downsample(df: pd.DataFrame,
               default_nr_samples: int,
               samples_per_label: dict = None,
               random_state: int = None) -> pd.DataFrame:
    """
    Downsamples the given DataFrame to contain at most `default_nr_samples` per instance of label.

    :param df: Input DataFrame.
    :param default_nr_samples: Default number of samples per label.
    :param samples_per_label: Number of samples for specific labels.
    :param random_state: Random state.
    :return: The downsampled DataFrame.
    """
    if samples_per_label is None:
        samples_per_label = {}

    sample_dict = create_sample_dict(df, default_nr_samples, samples_per_label)
    return pd.concat([df[df.label_cat == l].sample(n=n, random_state=random_state) for l, n in sample_dict.items()])
