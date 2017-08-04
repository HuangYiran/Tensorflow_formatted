import tensorflow as tf
import numpy as np

tf.app.flags.DEFINE_string("data", "mnist", "the name of the data")
tf.app.flags.DEFINE_string("model", "mlp_dropout_mnist", "the name of the model")
tf.app.flags.DEFINE_string("checkpoint_dir", "./", "the path of the checkpoint")
tf.app.flags.DEFINE_integer("batch_size", 100, "the size of the batch")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")

FLAGS = tf.app.flags.FLAGS
def starter():
    """
    only for test, load the model directly
    """
