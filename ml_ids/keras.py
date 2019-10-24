import numpy as np
import gc
from tensorflow import keras
from tensorflow.keras import callbacks
from sklearn.metrics import average_precision_score

K = keras.backend


class AveragePrecisionScoreMetric(callbacks.Callback):
    """
    Keras callback calculating the average precision score for a given validation dataset using the
    `average_precision_score` metric from Scikit-learn.
    """

    def __init__(self, X_val, y_val, batch_size=4096):
        super(AveragePrecisionScoreMetric, self).__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size

    def get_precision_score(self):
        preds = self.model.predict(self.X_val, batch_size=self.batch_size)
        # reduces memory consumed by a memory leak in `model.predict()` of Tensorflow 2
        # https://github.com/tensorflow/tensorflow/issues/33009
        gc.collect()
        mse = np.mean(np.power(self.X_val - preds, 2), axis=1)
        return average_precision_score(self.y_val, mse)

    def on_epoch_end(self, epoch, logs):
        auprc = self.get_precision_score()
        logs['val_auprc'] = auprc
        print(' - val_auprc: {0:.4f}'.format(auprc))


class OneCycleScheduler(callbacks.Callback):
    """
    Keras callback implementing a one-cycle learning-rate scheduler.
    Provided by https://github.com/ageron/handson-ml2/blob/master/11_training_deep_neural_networks.ipynb.
    """

    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0

    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (iter2 - self.iteration)
                / (iter2 - iter1) + rate1)

    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
            rate = max(rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)


# taken from
# https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))
