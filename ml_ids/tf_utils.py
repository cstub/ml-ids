"""
Utility functions for TensorFlow.
"""
import tensorflow as tf


def enable_gpu_memory_growth():
    """
    Enables the experimental setting `allow_memory_growth` for GPU devices

    :return: None
    """
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
