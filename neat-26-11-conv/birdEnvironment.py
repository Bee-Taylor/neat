import random
from os import listdir
from PIL import Image
import numpy as np
from copy import deepcopy as dc
import time
import dask.delayed as dd
from numba import cuda
from matplotlib import pyplot as plt

from dask.distributed import get_client


def convert_truth_row_to_numbers(row):
    return list(map(lambda x: 1 if x else 0, row))


def open_image_inner(species, id):
    return np.array(Image.open("birds/train/" + species + "/" + id))


def open_images(species):
    return [open_image_inner(species, id) for id in
            listdir("birds/train/" + species)]


def open_two_images(species):
    ids = random.sample(listdir("birds/train/" + species), 2)
    return [open_image_inner(species, id) for id in ids]


def open_images_multiple(species_list):
    filenames = []
    min = None
    for species in species_list:
        filenames.append(listdir("birds/train/" + species))
        if min is None or len(filenames[-1]) < min:
            min = len(filenames[-1])
    min //= 2
    to_return = np.zeros((len(species_list), min, 3, 224, 224))
    tmp = np.zeros((len(species_list), min, 224, 224, 3))
    for species_index in range(len(species_list)):
        for filename in range(min):
            tmp[species_index][filename] = np.array(Image.open("birds/train/" + species_list[species_index] + "/" +
                                    filenames[species_index][filename]))
    threadsperblock = 64
    blockspergrid = (tmp.size + (threadsperblock - 1)) // threadsperblock
    convert_image_to_runnable[blockspergrid, threadsperblock](tmp, to_return)
    return to_return

def open_shapes():
    to_return = np.zeros((4, 1, 3, 224, 224))
    tmp = np.zeros((4, 1, 224, 224, 3))
    for i in range(4):
        tmp[i][0] = np.array(Image.open("shapes/" + str(i) + ".jpg"))
    threadsperblock = 64
    blockspergrid = (tmp.size + (threadsperblock - 1)) // threadsperblock
    convert_image_to_runnable[blockspergrid, threadsperblock](tmp, to_return)
    return to_return

@cuda.jit
def convert_image_to_runnable(tmp, to_return):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    pos = tx + ty * bw

    species = pos // tmp[0].size
    pos_in_species = pos % tmp[0].size

    example = pos_in_species // tmp[0][0].size
    pos_in_example = pos_in_species % tmp[0][0].size

    y = pos_in_example // tmp[0][0][0].size
    x = (pos_in_example % tmp[0][0][0].size) // 3
    colour = pos % 3

    if y >= 224 or x >= 224 or example >= tmp.shape[1] or species >= tmp.shape[0]:
        return

    to_return[species][example][colour][y][x] = tmp[species][example][y][x][colour] / 128 - 1


def evaluate_agent_on_all(agent, examples):
    fitness = 0
    total = 0
    i = 0
    number_of_birds = examples.shape[1]
    correct = []
    outputs = []
    for bird in examples:
        for a in random.sample(list(range(examples.shape[1])), number_of_birds):
            inputs = np.zeros(examples.shape[2:])
            for colour in range(bird[a].shape[0]):
                inputs[colour] = bird[a][colour]
            outputs.append(agent.run(inputs))
            total += 1
            correct.append(i)
        i += 1
    for o in range(len(correct)):
        if np.argmax(outputs[o]) == correct[o]:
            fitness += 100
    return fitness / total


def evaluate_agent_non_learning(agent, examples, show=False, percentage=False):
    fitness = float(0)
    total = 0
    i = 0
    number_of_birds = examples.shape[1]
    inputs = np.zeros((examples.shape[0]*number_of_birds,  *examples.shape[2:]))
    correct = []
    for bird in examples:
        for a in random.sample(list(range(examples.shape[1])), number_of_birds):
            for colour in range(bird[a].shape[0]):
                inputs[total][colour] = bird[a][colour]
            total += 1
            correct.append(i)
        i += 1
    outputs = []
    for i in range(0, len(correct)):
        outputs.append(agent.run(inputs[i], show_as_images = show))
    #outputs = agent.run(inputs, show_as_images=show)
    prev_correct = -1
    for o in range(len(correct)):
        if percentage:
            if outputs[o].index(max(outputs[o])) == correct[o]:
                fitness += 100
        else:
            fitness += outputs[o][correct[o]]
            print(outputs[o], end = ", ")
            print(correct[o], end = ", ")
            print(outputs[o][correct[o]])
            tmp = inputs[o][0]
            plt.imshow(tmp, cmap = "gray")
            plt.show()
            prev_correct = correct[o]
            input()

    return fitness / total

def demonstrate(agent, examples):
    inputs = examples[random.randint(0, examples.shape[0] - 1)][random.randint(0, examples.shape[1] - 1)]
    print(agent.run(inputs))


if __name__ == "__main__":
    x = open_shapes()
    print(x)
