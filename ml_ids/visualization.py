import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.ticker import MaxNLocator


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

    fig, ax = plt.subplots(figsize=size)
    sns.distplot(transform(pred_train.rec_error.values), hist=False, ax=ax, label='Train Benign')
    sns.distplot(transform(pred_val[pred_val.y_true == 0].rec_error.values), hist=False, ax=ax,
                 label='Validation Benign')
    sns.distplot(transform(pred_val[pred_val.y_true == 1].rec_error.values), hist=False, ax=ax,
                 label='Validation Attack')
    ax.axvline(transform(threshold), color='red', linestyle='--')
    ax.legend()
