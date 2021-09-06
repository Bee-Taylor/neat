from os import listdir
import numpy as np
from PIL import Image
import random

class Omniglot:
    def __init__(self, characters): # characters: ndarray, shape: (2, n, x, y)
        self.characters = characters

    def run(self, individual):
        fitness = 0
        outputs_before = individual.run(self.characters[1])
        individual.run(self.characters[0], example = True)
        outputs = individual.run(self.characters[1])
        for o in range(self.characters.shape[1]):
            fitness += outputs[o][o] / self.characters.shape[1]
        return fitness + np.linalg.norm(outputs_before - outputs) / 100


def evaluate_agent(ind, omni):
    return omni.run(ind)

def openAll():
    languages = listdir("omniglot/archive/images_background/")
    to_return = np.zeros((964, 20, 1, 105, 105))
    indexes = [0, 0]
    for language in languages:
        characters = listdir("omniglot/archive/images_background/" + language)
        for character in characters:
            examples = listdir("omniglot/archive/images_background/" + language + "/" + character)
            for example in examples:
                to_return[indexes[0]][indexes[1]][0] = np.array(Image.open("omniglot/archive/images_background/" + language + "/" +
                                                         character + "/" + example))
                indexes[1] += 1
            indexes[1] = 0
            indexes[0] += 1
    return to_return

def make_new_omni(characters, n):
    example_indexes = [random.sample(list(range(characters.shape[1])), 2), random.sample(list(range(characters.shape[0])), n)]
    new_chars = np.zeros((2, n, 1, 105, 105))
    c = 0
    for char in example_indexes[1]:
        e = 0
        for example in example_indexes[0]:
            new_chars[e, c] = characters[char, example]
            e += 1
        c += 1
    omni = Omniglot(new_chars)
    return omni

if __name__ == "__main__":
    import time
    import conv_network_new

    st = time.time()
    x = openAll()
    omni = make_new_omni(x, 5)
    conv = conv_network_new.conv_network(105, 105, 1, 5, 3, True, 50)
    print(omni.run(conv))

