import tensorflow as tf
import numpy as np
import os

def load_mnist(data_dir):
    with open(os.path.join(data_dir, "train-images-idx3-ubyte.gz"), 'rb') as df:
        dt = np.dtype(np.uint32).newbyteorder('>')
        _ = np.frombuffer(df.read(4), dtype = np.uint32)
        num_images = np.frombuffer(df.read(4), dtype = np.uint32)[0]
        # frombuffer return array
        rows = np.frombuffer(df.read(4), dtype = np.uint32)[0]# the value i get here is terrible
        cols = np.frombuffer(df.read(4), dtype = np.uint32)[0]# ????? here use magic number
        buf = df.read(28*28*6000)
        data = np.frombuffer(buf, dtype = np.uint8)
        trX = data.reshape(-1, 28, 28, 1)

    with open(os.path.join(data_dir, "train-labels-idx1-ubyte.gz"), 'rb') as df:
        dt = np.dtype(np.uint32).newbyteorder('>')
        _ = np.frombuffer(df.read(4), dtype = dt)[0]
        num_labels = np.frombuffer(df.read(4), dtype = dt)[0]
        data = np.frombuffer(df.read(6000), dtype = np.uint8)
        trY = data
        print(trY)
    y_vec = np.zeros((len(trY), 10), dtype = np.float)
    for i, label in enumerate(trY):
        y_vec[i][trY[i]] = 1.0
    
    return trX/255, y_vec
"""
获取的y数据大于10，表示黑人问号脸。

"""
