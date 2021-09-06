import os
from os import listdir
from PIL import Image
import numpy as np
import random
from birdEnvironment import convert_image_to_runnable
from numba import cuda
from matplotlib import pyplot as plt

def openSample():
    names = listdir("dogsandcats/train/train/")
    names_split = [list(filter(lambda x: "cat" in x, names)), list(filter(lambda x: "dog" in x, names))]
    preprocessed_images = np.zeros((2, 50, 100 , 100, 3))
    to_return = np.zeros((2, 50, 3, 100, 100))
    to_return = cuda.to_device(to_return)
    for species in [0,1]:
        counter = 0
        for individual in random.sample(range(len(names_split[species])), 50):
            tmp = Image.open("dogsandcats/train/train/" + names_split[species][individual])
            tmp = tmp.resize((100, 100))
            preprocessed_images[species][counter] = np.array(tmp)
            counter += 1


    threadsperblock = 64
    blockspergrid = (preprocessed_images.size + (threadsperblock - 1)) // threadsperblock
    convert_image_to_runnable[blockspergrid, threadsperblock](preprocessed_images, to_return)
    return to_return


def replaceOne(pics):
    pics[0][:-1] = pics[0][1:]
    pics[1][:-1] = pics[1][1:]
    names = listdir("dogsandcats/train/train/")
    names_split = [list(filter(lambda x: "cat" in x, names)), list(filter(lambda x: "dog" in x, names))]
    preprocessed_images = np.zeros((2, 1, 100, 100, 3))
    to_return = np.zeros((2, 1, 3, 100, 100))
    for species in [0,1]:
        counter = 0
        for individual in random.sample(range(len(names_split[species])), 1):
            tmp = Image.open("dogsandcats/train/train/" + names_split[species][individual])
            tmp = tmp.resize((100, 100))
            preprocessed_images[species][counter] = np.array(tmp)
            counter += 1
    threadsperblock = 64
    blockspergrid = (preprocessed_images.size + (threadsperblock - 1)) // threadsperblock
    convert_image_to_runnable[blockspergrid, threadsperblock](preprocessed_images, to_return)
    pics[0][-1] = to_return[0][0]
    pics[1][-1] = to_return[1][-1]



if __name__ == "__main__":
    pics = openSample()
    print(pics)
