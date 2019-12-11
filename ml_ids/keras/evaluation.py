"""
Utility functions to evaluate Keras models.
"""
PREDICT_BATCH_SIZE = 16384


def evaluate_model(model, X_train, y_train, X_val, y_val, metric_title):
    """
    Prints the performance metrics of a Keras model by invoking the `evaluate` function of the model on the training
    and validation dataset.

    :param model: Keras model.
    :param X_train: Predictor variables of the training dataset.
    :param y_train: Target labels of the training dataset.
    :param X_val: Predictor variables of the validation dataset.
    :param y_val: Target labels of the validation dataset.
    :param metric_title: Title of the metrics.
    :return: None
    """
    print('Evaluation:')
    print('===========')
    print('       {}'.format(metric_title))
    print('Train: {}'.format(model.evaluate(X_train, y_train, batch_size=PREDICT_BATCH_SIZE, verbose=0)))
    print('Val:   {}'.format(model.evaluate(X_val, y_val, batch_size=PREDICT_BATCH_SIZE, verbose=0)))
