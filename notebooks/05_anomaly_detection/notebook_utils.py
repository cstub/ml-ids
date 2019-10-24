import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, average_precision_score, roc_auc_score, precision_recall_curve, \
    roc_curve
from IPython.display import display
from ml_ids.visualization import plot_confusion_matrix


def predict(model, X, y):
    preds = model.predict(X, batch_size=8196)
    mse = np.mean(np.power(X - preds, 2), axis=1)

    return pd.DataFrame({'y_true': y, 'rec_error': mse})


def evaluate_pr_roc(pred):
    pr_auc = average_precision_score(pred.y_true, pred.rec_error)
    roc_auc = roc_auc_score(pred.y_true, pred.rec_error)
    return pr_auc, roc_auc


def plot_evaluation_curves(pred):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    precisions, recalls, thresholds = precision_recall_curve(pred.y_true, pred.rec_error)
    fpr, tpr, _ = roc_curve(pred.y_true, pred.rec_error)
    pr_auc, roc_auc = evaluate_pr_roc(pred)

    # plot precision / recall curve
    ax1.plot(recalls, precisions, label='auc={}'.format(pr_auc))
    ax1.set_title('Precision / Recall Curve')
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.legend(loc='lower right')

    # plot ROC curve
    ax2.plot(fpr, tpr, label='auc={}'.format(roc_auc))
    ax2.set_title('ROC Curve')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_xlabel("False Positive Rate")
    ax2.legend(loc='lower right')


def plot_pr_threshold_curves(pred, pr_plot_lim=[0, 1]):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(20, 8))

    precisions, recalls, thresholds = precision_recall_curve(pred.y_true, pred.rec_error)

    # plot precision / recall for different thresholds
    ax1.plot(thresholds, precisions[:-1], label="Precision")
    ax1.plot(thresholds, recalls[:-1], label="Recall")
    ax1.set_title('Precision / Recall of different thresholds')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Precision / Recall')
    ax1.legend(loc='lower right')

    # plot precision / recall for different thresholds
    ax2.plot(thresholds, precisions[:-1], label="Precision")
    ax2.plot(thresholds, recalls[:-1], label="Recall")
    ax2.set_title('Precision / Recall of different thresholds')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Precision / Recall')
    ax2.set_xlim(pr_plot_lim)
    ax2.legend(loc='lower right')


def best_precision_for_target_recall(pred, target_recall):
    precisions, recalls, thresholds = precision_recall_curve(pred.y_true, pred.rec_error)
    return thresholds[np.argmin(recalls >= target_recall)]


def get_misclassifications(y, pred_binary):
    misclassifications = y[y.label_is_attack != pred_binary]

    mc_df = pd.merge(pd.DataFrame({'misclassified': misclassifications.label.value_counts()}),
                     pd.DataFrame({'total': y.label.value_counts()}),
                     how='left', left_index=True, right_index=True)
    mc_df['percent_misclassified'] = mc_df.apply(lambda x: x[0] / x[1], axis=1)
    return mc_df.sort_values('percent_misclassified', ascending=False)


def print_performance(y, pred, threshold):
    pred_binary = (pred.rec_error >= threshold).astype('int')

    print('Classification Report:')
    print('======================')
    print(classification_report(pred.y_true, pred_binary))

    print('Confusion Matrix:')
    print('=================')
    plot_confusion_matrix(pred.y_true, pred_binary, np.array(['Benign', 'Attack']), size=(5, 5))
    plt.show()

    print('Misclassifications by attack category:')
    print('======================================')
    mc_df = get_misclassifications(y, pred_binary)
    display(mc_df)


def filter_benign(X, y):
    return X[y.label_is_attack == 0]
