import tensorflow as tf
import numpy as np

"""
basic layer type, include:
    linear layer
    nonlinear layer
    conv2d layer
    deconv2d layer
    ...
visual utils are not jet used 
"""

def conv2d(input_, image_h, image_w, output_dim, stride_h, stride_w, stride_c = 1, stddev = 0.02, padding = 'SAME', use_cudnn_on_gpu = None, data_format = None, name = None, with_w = False):
    """
    becaause it's 2d convolution and in_channel = the number of channels from last layer: output_dim is the number of the filter
    w for fliter [image_h, image_w, in_channel, out_channel]
    in_channel is the same as image_c
    """
    image_c = input_.get_shape()[-1]
    with tf.variable_scope(name or 'conv2d'):
        w = tf.get_variable('w',
                [image_h, image_w, image_c, output_dim],
                initializer = tf.truncated_normal_initializer(stddev = stddev))
        biases = tf.get_variable('b', [output_dim],
                initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, [1, stride_h, stride_w, 1], padding = padding,
                use_cudnn_on_gpu = use_cudnn_on_gpu,
                data_format = data_format)
        #conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        if with_w:
            return conv, w, biases
        else:
            return conv

def conv2d_transpose(input_, image_h, image_w, output_shape, stride_h, stride_w, stride_c = 1, stddev = 0.02, padding = 'SAME', data_format = 'NHWC', name = None, with_w = False):
    """
    to get a large image from a small image through the filter.
    but it's not the averse of the conv stridly 
    filter: [height, width, otput_channel, input_channel]
    usually the first dimension of the output_shape is batch_size, but i am not sure here
    """
    image_c = output_shape[-1]
    with tf.variable_scope(name or 'conv2d_transpose'):
        w = tf.get_variable('w',
                [image_h, image_w, image_c, input_.getshape()[-1]],
                initializer = tf.truncated_normal_initializer(stddev = stddev))
        biases = tf.get_variable('b', [input_.get_shape()[-1]], 
                initializer = tf.constant_initializer(0.0))
        conv_transpose = tf.nn.conv2d.transpose(input_, w, output_shape,
                [1, stride_h, stride_w, 1],
                padding = padding,
                data_format = data_format)
        conv_transpose = tf.reshape(tf.nn.bias_add(conv_transpose, biases), conv_transpose.get_shape())
        if with_w:
            return conv_transpose, w, biases
        else:
            return conv_transpose

def linear(input_, output_dim, stddev = 0.02, name = None, with_w = False):
    input_dim = input_.get_shape()[-1]
    with tf.variables_scope(name or "linear"):
        w = tf.get_variable('w', [input_dim, output_dim],
                initializer = tf.truncated_normal_initializer(stddev = staddev))
        biases = tf.get_variable('b', [output_dim],
                initializer = tf.constant_initializer(0.0))
        if with_w:
            return tf.matmul(input_, w) + biases, w, biases
        else:
            return tf.matmul(input_, w) + biases

def nonlinear(input_, output_dim, stddev = 0.02, activation_function = tf.nn.relu, name = None, with_w = False):
    """
    activation_function is a function parameter. so i can use self-defined function here
    """
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name or "nonlinear"):
        w = tf.get_variable('w', [input_dim, output_dim],
                initializer = tf.truncated_normal_initializer(stddev = stddev))
        biases = tf.get_variable('b', [output_dim],
                initializer = tf.constant_initializer(0.02))
        sum = tf.matmul(input_, w) + biases
        if with_w:
            return activation_function(sum), w, biases
        else:
            return activation_function(sum)
