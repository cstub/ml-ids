import pandas as pd
from ml_ids.data.metadata import LABEL_CAT_MAPPING
from ml_ids.transform.sampling import downsample
from ml_ids.visualization import print_binary_performance, plot_pr_curve
from sklearn.model_selection import train_test_split

LABEL_LOOKUP = dict((v, k) for k, v in LABEL_CAT_MAPPING.items())


def split_train_test(data,
                     nr_samples_per_attack,
                     nr_samples_benign,
                     test_size,
                     nr_samples_benign_rest=None,
                     random_state=None):
    train_data, test_data = (train_test_split(data,
                                              test_size=test_size,
                                              stratify=data.label_cat,
                                              random_state=random_state))

    train_data_benign = train_data[train_data.label_is_attack == 0]
    train_data_attack = train_data[train_data.label_is_attack == 1]

    train_data_attack = downsample(train_data_attack,
                                   default_nr_samples=nr_samples_per_attack,
                                   random_state=random_state)

    train_benign, train_benign_rest = train_test_split(train_data_benign,
                                                       train_size=nr_samples_benign,
                                                       random_state=random_state)

    train_benign_rest = train_benign_rest.sample(n=nr_samples_benign_rest) \
        if nr_samples_benign_rest else train_benign_rest

    return (pd.concat([train_benign, train_data_attack])
            .sample(frac=1, random_state=random_state)), test_data, train_benign_rest


def print_performance(y_true,
                      y_pred_proba,
                      label_is_attack_index=1,
                      decision_boundary=0.5,
                      average='weighted'):
    y_true_df = pd.DataFrame(data=y_true, columns=['label_cat', 'label_is_attack'])
    y_true_df['label'] = y_true_df.label_cat.apply(lambda x: LABEL_LOOKUP[x])

    y_true_binary = y_true[:, label_is_attack_index]
    y_pred_binary = (y_pred_proba >= decision_boundary).astype('int').reshape(-1)

    plot_pr_curve(y_true[:, label_is_attack_index], y_pred_proba.reshape(-1), average=average)
    print('')
    print_binary_performance(y_true_df, y_true_binary, y_pred_binary)
