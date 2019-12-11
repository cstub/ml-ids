"""
Utilities to create predictions given a Scikit-learn estimator and a dataset containing input features.
"""


def predict_proba_positive(clf, X):
    """
    Performs predictions for a binary classification task given a scikit-learn model.
    This function returns the class probability of the positive class.

    :param clf: Scikit-learn estimator.
    :param X: Dataset containing the samples.
    :return: numpy array containing the class probabilities of the positive class.
    """
    return clf.predict_proba(X)[:, 1].reshape(-1)


def predict_decision_boundary(clf, X, decision_boundary=0.5):
    """
    Performs predictions for a binary classification task given a scikit-learn model and a decision boundary.
    If the probability of a sample belonging to the positive class exceeds the decision boundary the positive label
    is assigned to the sample, otherwise the negative label is used.

    :param clf: Scikit-learn estimator.
    :param X: Dataset containing samples.
    :param decision_boundary: Decision boundary used to assign predictions to the positive class.
    :return: numpy array containing the binary predictions as one of the values {0, 1}.
    """
    pred = predict_proba_positive(clf, X)
    return (pred >= decision_boundary).astype('int')
