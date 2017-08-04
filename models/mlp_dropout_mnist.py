import tensorflow as tf
import numpy as np

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
    def __init__(self, sess,l1_num, l1_act = None,
            l2_num, l2_act = None,
            stddev = 0.02,
            train_dir, test_dir, checkpoint_dir,
            batch_size):
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
        self.X, self.Y = load_mnist(train_dir)

        # initialize counter
        self.counter = 0

        # build model
        model()

        # saver
        self.saver = tf.train.Saver(tf.global_variables())


    def model(self):
        # placeholder
        x = tf.placeholder(tf.float32, [None, 28*28])
        y = tf.placeholder(tf.float32, [None, 10])
        keep_prob = tf.placeholder(tf.float32)

        # model
        h1, w1, b1 = nonlinear(x, self.l1_num, stddev = stddev,
                activation_function = self.l1_act, name = 'h1: nonlinear', with_w = True)

        h2, w2, b2 = nonlinear(self.h1, self.l2_num, stddev = stddev,
                activation_function = self.l2_act, name = 'h2: nonlinear', with_w = True)
        h2_dropout = tf.nn.dropout(h2, keep_prob = keep_prob)

        out = nonlinear(h2, 10, stddev = stddev,
                activation_function = tf.nn.softmax, name = 'out: nonlinear', with_w = False)

        # cross_entropy
        self.cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = y, logits = out))

        # train 
        self.train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        # evaluation
        is_correct = tf.equail(tf.argmax(y, 1), tf.argmax(out, 1))
        is_correct_int = tf.cast(is_correct, tf.float32)
        self.accuracy = reduce_mean(is_correct_in)

        # visualisation
        with tf.name_scope("hidden layer 1"):
            variable_summaries(w1)
            variable_summaries(b1)
            tf.summary.histogram("h1", h1)
        with tf.name_scope("hidden layer 2"):
            variable_summaries(w2)
            variable_summaries(b2)
            tf.summary.histogram("h2", h2)
        with tf.name_scope("cross_entropy"):
            tf.summary.histogram("cross_entropy", cross_entropy)
        with tf.name_scope("accuracy"):
            tf.summary.histogram("accuracy", accuracy)
        self.merged = tf.summary.merge_all()

    def step(self, sess, input_data, input_labels, train_writer = None):
        summary, _ = sess.run([merged, self.train], feed_dict = {x: input_data, y: input_labels, keep_prob: 0.5})
        if train_writer:
            train_writer.add_summary(summary, counter)
        self.counter = self.counter + 1

    def evaluate(self, sess, input_data, input_labels):
        sess.run(self.accuracy, feed_dict = {x: input_data, y: input_labels, keep_prob: 1.0})

    def get_batch(self):
        # get the data according to the counter
        ab = self.counter * self.batch_size
        to = (self.counter + 1) * self.batch_size
        if ab > len(self.X):
            return None
        else if to > len(self.X):
            return self.X[ab:], self.Y[ab:]
        else:
            return self.X[ab: to], self.Y[ab: to]
