import pickle
import numpy as np
import random

def load_all():
    random.seed(69)
    to_return = np.zeros((10, 10, 3, 32, 32), dtype=float)
    counters = [0 for i in range(10)]
    filename = "CIFAR/cifar-10-batches-py/data_batch_1"
    with open(filename, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    r = list(range(1000))
    random.shuffle(r)

    for j in r:
        classification = dict[b"labels"][j]
        if counters[classification] < to_return.shape[1]:
            to_return[classification][counters[classification]][0] = (dict[b"data"][j][:1024].reshape((32, 32)) / 128) - 1
            to_return[classification][counters[classification]][1] = (dict[b"data"][j][1024:2048].reshape((32, 32)) / 128) - 1
            to_return[classification][counters[classification]][2] = (dict[b"data"][j][2048:].reshape((32, 32)) / 128) - 1
            counters[classification] += 1
    return to_return

if __name__ == "__main__":
    print(load_all())