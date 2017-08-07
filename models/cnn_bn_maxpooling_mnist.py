import tensorflow as tf
import numpy as np
import sys

sys.path.append('data_utils')
sys.path.append('model_utils')
sys.path.append('visual_utils')

from read_data import *
from layer_utils import *
from visual_utils import *

class cnn_bn_maxpooling_mnist(object):
    """
    2 cnn layers + 2 bn layers + 2 max pooling layers + 2 fc layers
    train the network with adam_optimizer through minimizing the loss
    use euqal to calcu the accuracy
    """
    def __init__(self, sess, f1_h = 5, f1_w = 5, f1_out = 32, f2_h = 5, f2_w  = 5, f2_out = 64, fc1_n = 1024, batch_size = 100):
        """
        load parameters, load data, initilize counter and buld model, set train_writer, initialize counter, set saver
        parameters include:
            session: sess
            net_topo: f1_h, f1_w, f1_out, f2_h, f2_w, f2_out, fc1_n
            file_dir: train_dir, test_dir
            batch_size: batch_size
        """
        # load parameter
        self.sess = sess
        self.f1_h = f1_h
        self.f1_w = f1_w
        self.f1_out = f1_out
        self.f2_h = f2_h
        self.f2_w = f2_w
        self.f2_out = f2_out
        self.fc1_n = fc1_n
        self.batch_size = batch_size


        # load data 
        import tensorflow.examples.tutorials.mnist.input_data as input_data
        self.mnist = input_data.read_data_sets("../MNIST_data/", one_hot = True)

        # initialize counter
        self.global_counter = 0

        # build model
        self.model()

        # set train_writer
        self.train_writer = tf.summary.FileWriter('./log/train', sess.graph)

        # set saver
        self.saver = tf.train.Saver(tf.global_variables())

    def model(self):
        # set placehodler
        self.x = tf.placeholder(tf.float32, [None, 28 * 28])
        x_ = tf.reshape(self.x, [-1, 28, 28, 1])
        self.y_ = tf.placeholder(tf.float32, [None, 10])

        # set topo
        c1, c1_w, c1_b = conv2d(x_, self.f1_h, self.f1_w, self.f1_out, stride_h = 1, stride_w = 1, with_w = True, name = 'conv1')
        c1_bn = tf.layers.batch_normalization(c1)
        c1_bn_mp = tf.nn.max_pool(c1_bn, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        c2, c2_w, c2_b = conv2d(c1_bn_mp, self.f2_h, self.f2_w, self.f2_out, stride_h = 1, stride_w= 1, with_w = True, name = 'conv2')
        c2_bn = tf.layers.batch_normalization(c2)
        c2_bn_mp = tf.nn.max_pool(c2_bn, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')
        c2_bn_mp_reshape = tf.reshape(c2_bn_mp, [-1, 7 * 7 * self.f2_out])
        fc1, fc1_w, fc1_b = nonlinear(c2_bn_mp_reshape, self.fc1_n, with_w = True, name = 'fc1')
        fc2, fc2_w, fc2_b = nonlinear(fc1, 10, with_w = True, name = 'fc2')

        # set cross entropy 
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.y_, logits = fc2))

        # set train
        self.train = tf.train.AdamOptimizer(1e-4).minimize(self.cross_entropy)

        # set accuracy
        is_equal = tf.equal(tf.argmax(self.y_, 1), tf.argmax(fc2, 1))
        self.accuracy = tf.reduce_mean(tf.cast(is_equal, tf.float32))

        # visualisation
        with tf.name_scope('conv1'):
            with tf.name_scope('w'):
                variable_summarises(c1_w)
            with tf.name_scope('b'):
                variable_summarises(c1_b)
            with tf.name_scope('c1_bn'):
                tf.summary.histogram('c1_bn', c1_bn)
            with tf.name_scope('c1_bn_mp'):
                tf.summary.histogram('c1_bn_mp', c1_bn_mp)
        with tf.name_scope('conv2'):
            with tf.name_scope('w'):
                variable_summarises(c2_w)
            with tf.name_scope('b'):
                variable_summarises(c2_b)
            with tf.name_scope('c2_bn'):
                tf.summary.histogram('c2_bn', c2_bn)
            with tf.name_scope('c2_bn_mp'):
                tf.summary.histogram('c2_bn_mp', c2_bn_mp)
        with tf.name_scope('fc1'):
            with tf.name_scope('w'):
                variable_summarises(fc1_w)
            with tf.name_scope('b'):
                variable_summarises(fc1_b)
            with tf.name_scope('out'):
                tf.summary.histogram('fc1', fc1)
        with tf.name_scope('fc2'):
            with tf.name_scope('w'):
                variable_summarises(fc2_w)
            with tf.name_scope('b'):
                variable_summarises(fc2_b)
            with tf.name_scope('out'):
                tf.summary.histogram('fc2', fc2)
        self.merge = tf.summary.merge_all()

    def step(self, sess, input_data, input_labels):
        summary, _ = sess.run([self.merge, self.train],
                feed_dict = {self.x: input_data, self.y_: input_labels})
        self.train_writer.add_summary(summary, self.global_counter)
        self.global_counter = self.global_counter + 1

    def evaluate(self, sess, input_data, input_labels):
        accuracy = sess.run(self.accuracy,
                feed_dict = {self.x: input_data, self.y_: input_labels})
        print(accuracy)

    def get_batch(self):
        return self.mnist.train.next_batch(self.batch_size)
