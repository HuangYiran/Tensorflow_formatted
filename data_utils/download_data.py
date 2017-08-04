import tensorflow.examples.tutorials.mnist.input_data as input_data
import tensorflow as tf
import numpy as np

import os
import gzip
import logging

DATA_DIR = "../../data/Tensorflow_formatted"
logging.basicConfig(filename = 'logger_data_util.log', level = logging.INFO)

def mnist_download():
    MNIST_DIR = DATA_DIR + "/MNIST"
    train_images_path = MNIST_DIR + "/train-images-idx3-ubyte.gz"
    train_labels_path = MNIST_DIR + "/train-labels-idx1-ubyte.gz"
    test_images_path = MNIST_DIR + "/t10k-images-idx3-ubyte.gz"
    test_labels_path = MNIST_DIR + "/t10k-labels-idx3-ubyte.gz"

    #test weather the file exitsed or not
    if not os.path.exists(train_images_path):
        logging.info('mnist file does not exist, try to download the file and return the path...\n start dowdloading...')
        if not os.path.exists(MNIST_DIR):
            os.makedirs(MNIST_DIR)
        mnist = input_data.read_data_sets(MNIST_DIR, one_hot = True)
        logging.info('finish downloading.')
    else:
        logging.info('mnist file existed jet, return the path of the file')
    return (train_images_path, train_labels_path, test_images_path, test_labels_path)

def prepare_data():
    train_images_path, train_labels_path, test_images_path, test_labels_path = mnist_download()
    return (train_images_path, train_labels_path, test_images_path, test_labels_path)

prepare_data()
