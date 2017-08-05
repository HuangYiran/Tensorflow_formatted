import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append('./models')
sys.path.append('./data_utils')
from mlp_dropout_mnist import *
from download_data import *


tf.app.flags.DEFINE_string("data", "mnist", "the name of the data")
tf.app.flags.DEFINE_string("model", "mlp_dropout_mnist", "the name of the model")
tf.app.flags.DEFINE_string("checkpoint_dir", "./", "the path of the checkpoint")
tf.app.flags.DEFINE_string("train_dir", None, "the path of the training data")
tf.app.flags.DEFINE_string("test_dir", None, "the path of the testing data")
tf.app.flags.DEFINE_integer("batch_size", 100, "the size of the batch")
tf.app.flags.DEFINE_integer("l1_num", 300, "the number of paceptron in layer one")
tf.app.flags.DEFINE_integer("l2_num", 500, "the number of paceptron in layer two")
tf.app.flags.DEFINE_float("learning_rate", 1e-4, "Learning rate")

FLAGS = tf.app.flags.FLAGS

train_images_path, train_labels_path, test_images_path, test_labels_path = mnist_download()
train_dir = os.path.dirname(train_images_path)
test_dir = os.path.dirname(test_images_path)

with tf.Session() as sess:
    model = mlp_dropout_mnist(sess, FLAGS.l1_num, FLAGS.l2_num,
            train_dir = train_dir, test_dir = test_dir, checkpoint_dir = FLAGS.checkpoint_dir,
            batch_size = FLAGS.batch_size)
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        input_data, input_label = model.get_batch()
        model.step(sess, input_data, input_label)
        if i % 100 == 0:
            model.evaluate(sess, input_data, input_label)

"""
def starter():
    only for test, load the model directly

def build_model(sess):
    model = mlp_dropout_mnist(sess, FLAGS.l1_num, FLAGS.l2_num,
            train_dir = train_dir, test_dir = test_dir, checkpoint_dir = FLAGS.checkpoint_dir,
            batch_size = FLAGS.batch_size)
    ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" %(ckpt.model_checkpoint_path))
        model.saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print("Initialize the parameters")
        sess.run(tf.global_variables_initializer())
    return model

def train():
    if not FLAGS.train_dir:
"""

