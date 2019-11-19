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
