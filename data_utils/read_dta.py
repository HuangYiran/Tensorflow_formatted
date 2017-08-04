import tensorflow as tf
import numpy as np

def load_mnist(data_dir):
    fd = open(os.path.join(data_dir, "train-images-idx3-ubyte"))
    loaded = np.fromfile(file = fd, dtype = np.unit8)
    trX = loaded[16:].reshape((-1, 28, 28, 1)).astype(np.float)
    fd.close()

    fd = open(os.path.join(data_dir, "train-labels-idx1-ubyte"))
    loaded = np.fromfile(file = fd, dtype = np.unit8)
    trY = loaded[8:].reshape((-1)).astype(np.float)
    fd.close()

    y_vec = np.zeros((len(trY), 10), dtype = np.float)
    for i, label in enumerate(trY):
        y_vec[i, trY[i]] = 1.0
    
    return trX/255, y_vec
