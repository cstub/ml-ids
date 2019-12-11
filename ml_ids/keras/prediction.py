"""
Utility functions to create predictions using Keras models.
"""
PREDICT_BATCH_SIZE = 16384


def predict(model, X, decision_boundary=0.5):
    """
    Performs predictions for a binary classification task given a Keras model and a decision boundary.
    If the probability of a sample belonging to the positive class exceeds the decision boundary the positive label
    is assigned to the sample, otherwise the negative label is used.

    :param model: Keras model.
    :param X: Dataset containing samples.
    :param decision_boundary: Decision boundary used to assign predictions to the positive class.
    :return: numpy array containing the binary predictions as one of the values {0, 1}.
    """
    pred = model.predict(X, batch_size=PREDICT_BATCH_SIZE)
    return (pred >= decision_boundary).astype('int').reshape(-1)


def predict_proba(model, X):
    """
     Performs predictions for a binary classification task given a Keras model.
     This function returns the class probability of the positive class.

    :param model: Keras model.
    :param X: Dataset containing samples.
    :return: numpy array containing the class probabilities of the positive class.
    """
    return model.predict(X, batch_size=PREDICT_BATCH_SIZE).reshape(-1)
