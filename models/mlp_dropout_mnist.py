import tensorflow as tf
import numpy as np
import os
import sys

sys.path.append('data_utils')
sys.path.append('model_utils')
sys.path.append('visual_utils')

from read_data import *
from layer_utils import *
from visual_utils import *

class mlp_dropout_mnist(object):
    """
    two hidden layers, use cross_entropy to calcu the loss 
    train the network with adam_optimizer through minimize the loss.
    use equal to calcu the accuracy
    methods include:
        __init__()
        model()
        step()
        evaluate()
        get_batch()
    """
    def __init__(self, sess,l1_num,
            l2_num, 
            train_dir, test_dir, checkpoint_dir,
            batch_size,
            stddev = 0.02,
            l1_act = None,
            l2_act = None):
        """
        load parameters, load data and build model, initialize counter
        parameters include:
            sess: session
            file_dir: train_data, test_data, checkpoint
            network_topo: l1_num, l2_num(the number of paceptron in a layer)
            network_act_fun: l1_act=None, l2_act=None
            normal_variance: stddev=0.02
            batch_size: batch_size = 500
        """
        self.l1_num = l1_num
        self.l1_act = l1_act
        self.l2_num = l2_num
        self.l2_act = l2_act
        self.stddev = 0.02
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.checkpoint_dir = checkpoint_dir

        # load data
        #self.X, self.Y = load_mnist(train_dir)
        import tensorflow.examples.tutorials.mnist.input_data as input_data
        self.mnist = input_data.read_data_sets("../MNIST_data/", one_hot = True)

        # initialize counter
        self.counter = 0

        # build model
        self.model()

        # set train_writer
        self.train_writer = tf.summary.FileWriter('./log/train', sess.graph)

        # saver
        self.saver = tf.train.Saver(tf.global_variables())


    def model(self):
        # placeholder
        self.x = tf.placeholder(tf.float32, [None, 28*28])
        self.y = tf.placeholder(tf.float32, [None, 10])
        self.keep_prob = tf.placeholder(tf.float32)

        # model
        h1, w1, b1 = nonlinear(self.x, self.l1_num, stddev = self.stddev,
                name = 'h1_nonlinear', with_w = True)

        h2, w2, b2 = nonlinear(h1, self.l2_num, stddev = self.stddev,
                name = 'h2_nonlinear', with_w = True)
        h2_dropout = tf.nn.dropout(h2, keep_prob = self.keep_prob)

        out = nonlinear(h2, 10, stddev = self.stddev,
                activation_function = tf.nn.softmax, name = 'out_nonlinear', with_w = False)

        # cross_entropy
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y, logits = out))

        # train 
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # evaluation
        is_correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(out, 1))
        is_correct_int = tf.cast(is_correct, tf.float32)
        self.accuracy = tf.reduce_mean(is_correct_int)

        # visualisation
        with tf.name_scope("hidden_layer_1"):
            with tf.name_scope("w1"):
                variable_summarises(w1)
            with tf.name_scope("b1"):
                variable_summarises(b1)
            with tf.name_scope("h1_out"):
                tf.summary.histogram("h1", h1)
        with tf.name_scope("hidden_layer_2"):
            with tf.name_scope("w2"):
                variable_summarises(w2)
            with tf.name_scope("b2"):
                variable_summarises(b2)
            with tf.name_scope("h2_out"):
                tf.summary.histogram("h2", h2)
        with tf.name_scope("cross_entropy"):
            tf.summary.histogram("cross_entropy", self.cross_entropy)
        with tf.name_scope("accuracy"):
            tf.summary.histogram("accuracy", self.accuracy)
        self.merged = tf.summary.merge_all()

    def step(self, sess, input_data, input_labels):
        summary, _ = sess.run([self.merged, self.train], feed_dict = {self.x: input_data, self.y: input_labels, self.keep_prob: 0.5})
        self.train_writer.add_summary(summary, self.counter)
        self.counter = self.counter + 1

    def evaluate(self, sess, input_data, input_labels):
        print(sess.run(self.accuracy, feed_dict = {self.x: input_data, self.y: input_labels, self.keep_prob: 1.0}))

    def get_batch(self):
        # get the data according to the counter
        """
        ab = self.counter * self.batch_size
        to = (self.counter + 1) * self.batch_size
        if ab > len(self.X):
            return None
        elif to > len(self.X):
            return self.X[ab:], self.Y[ab:]
        else:
            return self.X[ab: to], self.Y[ab: to]
        """
        return self.mnist.train.next_batch(100)
