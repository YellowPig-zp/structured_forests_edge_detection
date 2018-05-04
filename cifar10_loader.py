import pickle
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    with open(file, "rb") as fo:
        batch_dict = pickle.load(fo, encoding="bytes")
    return batch_dict

def load(file):
    filepath = "./cifar-10-batches-py/"
    filepath += file
    batch_dict = unpickle(filepath)
    batch_data = batch_dict[b"data"]
    images, labels = [], batch_dict[b"labels"]
    for i in range(len(batch_data)):
        img = batch_data[i]
        blank_img = np.zeros([32, 32, 3], dtype="uint8")
        for j in range(3):
            blank_img[:, :, j] = img[32*32*j: 32*32*(j+1)].reshape((32, 32))
        images.append(blank_img)
    return images, labels
