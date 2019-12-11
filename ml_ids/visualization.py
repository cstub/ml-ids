"""
Visualization utilities for IPython Notebooks.
"""
# pylint: disable=import-error
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, classification_report, average_precision_score, precision_recall_curve
from IPython.display import display


def plot_hist(hist,
              metrics=None,
              y_lim=None,
              size=(8, 5),
              ax=None):
    """
    Plot a Keras history object.

    :param hist: The Keras history.
    :param metrics: A list of histories to plot.
    :param y_lim: Limits the y-axis.
    :param size: Size of the plot.
    :param ax: Axis to apply the plot.
    """
    if metrics is None:
        metrics = ['loss', 'val_loss']

    fig_size = size if not ax else None

    df = pd.DataFrame(hist.history)[metrics]
    df.plot(figsize=fig_size, ax=ax)

    gca = ax if ax else plt.gca()
    gca.xaxis.set_major_locator(MaxNLocator(integer=True))

    if y_lim:
        gca.set_ylim(y_lim)

    if ax:
        ax.grid(True)
    else:
        plt.grid(True)
        plt.show()


def plot_confusion_matrix(y_true,
                          y_pred,
                          classes=None,
                          size=(10, 10),
                          normalize=False,
                          title=None,
                          print_raw=False,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.

    :param y_true: True labels.
    :param y_pred: Predicted labels.
    :param classes: List of class names.
    :param size: Size of the plot.
    :param normalize: If True values of the confusion matrix will be normalized.
    :param title: Title of the plot.
    :param print_raw: If True the raw confusion matrix is printed.
    :param cmap: Color map
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    if print_raw:
        print(cm)

    fig, ax = plt.subplots(figsize=size)
    im = ax.matshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(title=title,
           ylabel='True label',
           xlabel='Predicted label')

    if classes is not None:
        x_labels = classes
        y_labels = classes

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=x_labels,
               yticklabels=y_labels)

    plt.margins(2)
    ax.tick_params(axis="x", bottom=True, labelbottom=True, top=False, labeltop=False, rotation=45)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def identity(x):
    """
    Identity function.
    """
    return x


def plot_threshold(pred_train, pred_val, threshold, size=(15, 5), transform=identity):
    """
    Plots the reconstruction errors of training and test samples and displays the classification threshold.

    :param pred_train: Predictions of training samples.
    :param pred_val: Predictions of validation samples.
    :param threshold: Classification threshold.
    :param size: Size of the plot.
    :param transform: Value transformation.
    """
    _, ax = plt.subplots(figsize=size)
    sns.distplot(transform(pred_train.rec_error.values), hist=False, ax=ax, label='Train Benign')
    sns.distplot(transform(pred_val[pred_val.y_true == 0].rec_error.values), hist=False, ax=ax,
                 label='Validation Benign')
    sns.distplot(transform(pred_val[pred_val.y_true == 1].rec_error.values), hist=False, ax=ax,
                 label='Validation Attack')
    ax.axvline(transform(threshold), color='red', linestyle='--')
    ax.legend()


def get_misclassifications(y, y_true, pred):
    """
    Calculates the misclassification rate for each label.

    :param y: Pandas DataFrame containing the target labels.
    :param y_true: True labels.
    :param pred: Predicted labels.
    :return: Pandas DataFrame containing the misclassification per label.
    """
    misclassifications = y[y_true != pred]

    mc_df = pd.merge(pd.DataFrame({'misclassified': misclassifications.label.value_counts()}),
                     pd.DataFrame({'total': y.label.value_counts()}),
                     how='left', left_index=True, right_index=True)
    mc_df['percent_misclassified'] = mc_df.apply(lambda x: x[0] / x[1], axis=1)
    return mc_df.sort_values('percent_misclassified', ascending=False)


def print_binary_performance(y, y_true, pred, print_misclassifications=True, digits=3):
    """
    Prints the performance of a binary classifier using
    - the classification report,
    - the confusion matrix and
    - the misclassification report.

    :param y: Pandas DataFrame containing the target labels (binary, categories).
    :param y_true: True labels.
    :param pred: Predicted labels.
    :param print_misclassifications: Binary indicator instructing that the misclassification report should be printed.
    :param digits: Number of digits used to print the classification report.
    :return: None
    """
    print('Classification Report:')
    print('======================')
    print(classification_report(y_true, pred, digits=digits))

    print('Confusion Matrix:')
    print('=================')
    plot_confusion_matrix(y_true, pred, np.array(['Benign', 'Attack']), size=(5, 5))
    plt.show()

    if print_misclassifications:
        print('Misclassifications by attack category:')
        print('======================================')
        mc_df = get_misclassifications(y, y_true, pred)
        display(mc_df)


def plot_pr_curve(y_true, y_score, size=(8, 5), average='weighted'):
    """
    Plots the precision-recall curve for a single estimator.

    :param y_true: True labels.
    :param y_score: Predicted probabilities.
    :param size: Size of the plot.
    :param average: Average parameter used for the calculation of the average precision score.
    :return: None
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score, average=average)

    plt.figure(figsize=size)
    plt.plot(recalls, precisions, label='auc={}'.format(pr_auc))
    plt.title('Precision / Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()

    print('Average PR Score {}'.format(pr_auc))


def plot_pr_curves(y_true, y_score_dict, size=(8, 5), average='weighted'):
    """
    Plots the precision-recall curve for a multiple estimators.

    :param y_true: True labels.
    :param y_score_dict: Dictionary containing the estimator name as keys and the predicted label probabilities
           as values.
    :param size: Size of the plot.
    :param average: Average parameter used for the calculation of the average precision score.
    :return: None
    """
    plt.figure(figsize=size)

    for name, y_score in y_score_dict.items():
        precisions, recalls, _ = precision_recall_curve(y_true, y_score)
        pr_auc = average_precision_score(y_true, y_score, average=average)
        plt.plot(recalls, precisions, label='{} (AUC={})'.format(name, pr_auc))

    plt.title('Precision / Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.show()


def plot_pr_threshold_curves(y_true, y_pred_score, size=(20, 8)):
    """
    Plots the precision-recall values for different probability thresholds.

    :param y_true: True labels.
    :param y_pred_score: Predicted probabilities.
    :param size: Size of the plot.
    :return: None
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_score)

    # plot precision / recall for different thresholds
    plt.figure(figsize=size)
    plt.plot(thresholds, precisions[:-1], label="Precision")
    plt.plot(thresholds, recalls[:-1], label="Recall")
    plt.title('Precision / Recall of different thresholds')
    plt.xlabel('Threshold')
    plt.ylabel('Precision / Recall')
    plt.legend(loc='lower right')
    plt.show()
