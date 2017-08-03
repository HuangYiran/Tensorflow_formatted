import tensorflow as tf
import numpy as np

"""
add some activation functions, which not appear in tensorflow.org:
    lReLu
"""

def lrelu(x, leaky = 0.2):
    return max(x, leaky*x)
