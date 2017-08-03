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

def conv2d(input_, output_dim, stride_h, stride_w, stride_c = 1, stddev = 0.02, padding = 'SAME', use_cudnn_on_gpu = None, data_format = None, name = None):
    """
    becaause it's 2d convolution: output_dim is the number of the filter
    ??should i use params image_h, image_w. In fact i can get it from input through input.get_size
    """
    image_h, image_w, image_c = input_.get_size()[1:]
    with tf.variables_scope(name or 'conv2d'):
        w = tf.get_variable('w',
                [image_h, image_w, image_c, output_dim],
                initializer = tf.truncated_normal_initializer(stddev = stddev))
        biases = tf.get_variable('b', [output_dim],
                initializer = tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, [1, stride_h, stride_w, 1], padding = padding,
                use_cudnn_on_gpu = use_cudnn_on_gpu,
                data_format = data_fromat)
        conv = tf.reshape(tf.nn.biases_add(conv, biases), conv.get_shape())
        return conv

def conv2d_transpose(input_, output_shape, stride_h, stride_w, stride_c = 1, stddev = 0.02, padding = 'SAME', data_format = 'NHWC', name = None):
    """
    to get a large image from a small image through the filter.
    but it's not the averse of the conv stridly 
    filter: [height, width, otput_channel, input_channel]
    usually the first dimension of the output_shape is batch_size, but i am not sure here
    """
    image_h, image_w, image_c = output_shape[-3:-1]
    with tf.variables_scope(name or 'conv2d_transpose'):
        w = tf.get_variable('w',
                [image_h, image_w, image_c, input_.getshape()[-1]],
                initializer = tf.truncated_normal_initializer(stddev = stddev))
        biases = tf.get_variable('b', [input_.get_shape()[-1]], 
                initializer = tf.constant_initializer(0.0))
        conv_transpose = tf.nn.conv2d.transpose(input_, w, output_shape,
                [1, stride_h, stride_w, 1],
                padding = padding,
                data_format = data_format)
        conv_transpose = tf.reshape(tf.nn.biases_add(conv_transpose, biases), conv_transpose.get_shape())
        return conv_transpose

def linear(input_, output_dim, stddev = 0.02, name = None):
    input_dim = input_.get_shape()[-1]
    with tf.variables_scope(name or "linear"):
        w = tf.get_variable('w', [input_dim, output_dim],
                initializer = tf.truncated_normal_initializer(stddev = staddev))
        biases = tf.get_variable('b', [output_dim],
                initializer = tf.constant_initializer(0.0))
        return tf.matmul(input_, w) + biases

def nonlinear(input_, output_dim, stddev = 0.02, activation_function = None, name = None):
    """
    activation_function is a function parameter. so i can use self-defined function here
    """
    input_dim = input_.get_shape()[-1]
    with tf.variable_scope(name or "nonlinear"):
        w = tf.get_variable('w', [input_dim, output_dim],
                initializer = tf.truncated_normal_initiallizer(stddev = stddev))
        biases = tf.get_variable('b', [output_dim],
                initializer = tf.constant_initializer(0.02))
        sum = tf.matmul(input_, w) + biases
        if activation_function:
            return tf.sigmoid(sum)
        else:
            return activation_function(sum)
