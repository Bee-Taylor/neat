from os import listdir
import numpy as np
from PIL import Image


def openAllCharacters():
    filenames_inner = []
    filenames = listdir("Latin/")
    min_count = None
    for f in filenames:
        filenames_inner.append(listdir("Latin/" + f))
        if min_count is None or len(filenames[-1]) < min_count:
            min_count = len(filenames[-1])
    images = np.zeros((len(filenames), min_count, 105, 105))
    for f in range(len((filenames))):
        for f1 in range(min_count):
            images[f][f1] = np.array(Image.open("Latin/" + filenames[f] + "/" + filenames_inner[f][f1]))
    images_converted = np.zeros((len(filenames), min_count, 1, 105, 105))
    for f in range(26):
        for i in range(min_count):
            for y in range(105):
                for x in range(105):
                    images_converted[f][i][0][y][x] = images[f][i][y][x]
    images_converted -= 1
    images_converted *= -1
    return images_converted



if __name__ == "__main__":
    openAllCharacters()