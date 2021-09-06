import math
import time

from numba.cuda.cudadrv.driver import CudaAPIError

import ArtificialNetwork
import random
from copy import deepcopy as dc
import numpy as np
from PIL import Image
from numba import cuda
from matplotlib import pyplot as plt
import GLCM
import CGP


class Max_Pool_Section:
    def __init__(self, same):
        self.same = same
        pass

    def run(self, panes):
        if self.same:
            outputs_new = np.zeros(panes.shape)
        else:
            outputs_new = np.zeros((panes.shape[0], int((panes.shape[1] - 1) / 2), int((panes.shape[2] - 1) / 2)))
        threadsperblock = 64
        blockspergrid = (outputs_new.size + threadsperblock - 1) // threadsperblock
        max_pool_gpu[blockspergrid, threadsperblock](panes, outputs_new, self.same)
        return outputs_new

    def __repr__(self):
        return '{"max_pool": true}'


def add_at_pos_each_row(array_2d, vector, pos):
    to_return = np.zeros((array_2d.shape[0], array_2d.shape[1] + 1))
    for row in range(array_2d.shape[0]):
        to_return[row][:pos] = array_2d[row][:pos]
        to_return[row][pos] = vector[row]
        to_return[row][pos + 1:] = array_2d[row][pos:]
    return to_return


# outputs = (pane_count, new_width, new_height)
# panes = (pane_count, width, height
@cuda.jit
def max_pool_gpu(channels, outputs, same):  # channel, y, x
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    pos = tx + ty * bw

    target_channel = pos // outputs[0].size

    pos2 = pos % outputs[0].size
    if same:
        y_output = pos2 // channels.shape[1]
        y_input = y_output
        x_output = pos2 % channels.shape[2]
        x_input = x_output
    else:
        y_output = pos2 // int((channels.shape[1] - 1) / 2)
        x_output = pos2 % int((channels.shape[1] - 1) / 2)
        y_input = y_output * 2 + 1
        x_input = x_output * 2 + 1

    if target_channel >= channels.shape[0] or y_input >= channels.shape[1] or x_input >= channels.shape[2]:
        return
    max = channels[target_channel][y_input - 1][x_input - 1]
    if channels[target_channel][y_input - 1][x_input] > max:
        max = channels[target_channel][y_input - 1][x_input]
    if x_input < len(channels[target_channel][0]) - 1 and channels[target_channel][y_input - 1][x_input + 1] > max:
        max = channels[target_channel][y_input - 1][x_input + 1]

    if channels[target_channel][y_input][x_input - 1] > max:
        max = channels[target_channel][y_input][x_input - 1]

    if channels[target_channel][y_input][x_input] > max:
        max = channels[target_channel][y_input][x_input]
    if x_input < len(channels[target_channel][0]) - 1 and channels[target_channel][y_input][x_input + 1] > max:
        max = channels[target_channel][y_input][x_input + 1]

    if y_input < len(channels[target_channel]) - 1 and channels[target_channel][y_input + 1][x_input - 1] > max:
        max = channels[target_channel][y_input + 1][x_input - 1]
    if y_input < len(channels[target_channel]) - 1 and channels[target_channel][y_input + 1][x_input] > max:
        max = channels[target_channel][y_input + 1][x_input]
    if y_input < len(channels[target_channel]) - 1 and x_input < len(channels[target_channel][0]) - 1 and \
            channels[target_channel][y_input + 1][x_input + 1] > max:
        max = channels[target_channel][y_input + 1][x_input + 1]
    outputs[target_channel][y_output][x_output] = max


class Inception:
    def __init__(self, width, height, depth_in, depth_out, final_layer=False):
        self.width = width
        self.height = height
        self.depth_in = depth_in
        self.depth_out = depth_out
        self.final_layer = final_layer
        self.previous_mutation_vector = []
        self.mutation_sd = 0.1
        if final_layer:
            self.C1F1 = np.zeros((depth_out, depth_in + 1))
        else:
            self.C1F1 = np.zeros((depth_out // 5, depth_in + 1))
        self.C3F1 = np.zeros((depth_in, depth_in + 1))
        self.C5F1 = np.zeros((depth_in, depth_in + 1))
        self.C7F1 = np.zeros((depth_in, depth_in + 1))
        for d1 in range(depth_in):
            for d2 in range(depth_in + 1):
                if d1 < self.C1F1.shape[0]:
                    self.C1F1[d1, d2] = random.gauss(0, 1)
                self.C3F1[d1, d2] = random.gauss(0, 1)
                self.C5F1[d1, d2] = random.gauss(0, 1)
                self.C7F1[d1, d2] = random.gauss(0, 1)

        self.C3F2 = np.zeros((depth_out // 5 + depth_out % 5, depth_in * 9 + 1))
        for d1 in range(depth_out // 5 + depth_out % 5):
            for d2 in range(depth_in * 9 + 1):
                self.C3F2[d1, d2] = random.gauss(0, 1)
        self.C5F2 = np.zeros((depth_out // 5, depth_in * 25 + 1))
        for d1 in range(depth_out // 5):
            for d2 in range(depth_in * 25 + 1):
                self.C5F2[d1, d2] = random.gauss(0, 1)
        self.C7F2 = np.zeros((depth_out // 5, depth_in * 49 + 1))
        for d1 in range(depth_out // 5):
            for d2 in range(depth_in * 49 + 1):
                self.C7F2[d1, d2] = random.gauss(0, 1)
        self.CMF2 = np.zeros((depth_out // 5, depth_in + 1))
        for d1 in range(depth_out // 5):
            for d2 in range(depth_in + 1):
                self.CMF2[d1, d2] = random.gauss(0, 1)

    def mutate(self, to_add):
        new_mutation_vector = []
        mut_vector_index = 0
        mutate_sporadic = random.choice([True, False])
        # mutate weights
        for to_mutate in [self.C1F1, self.C3F1, self.C5F1, self.C7F1, self.C3F2, self.C5F2, self.C7F2, self.CMF2]:
            for f in range(to_mutate.shape[0]):
                for w in range(to_mutate.shape[1]):
                    if mutate_sporadic or random.random() < 0.05:
                        if mut_vector_index >= len(self.previous_mutation_vector):
                            tmp = random.gauss(0, 0.1)
                            to_mutate[f][w] += tmp
                            new_mutation_vector.append(tmp)
                        else:
                            tmp = self.previous_mutation_vector[mut_vector_index] + random.gauss(0, self.mutation_sd)
                            to_mutate[f][w] += tmp
                            new_mutation_vector.append(tmp)
                        mut_vector_index += 1
                    else:
                        new_mutation_vector.append(0)
        self.previous_mutation_vector = new_mutation_vector

        if to_add is not None and to_add > 0:
            self.depth_in = self.depth_in + 1
            vector_C1F1 = np.zeros((self.C1F1.shape[0], 1))
            vector_C3F1 = np.zeros((self.C3F1.shape[0], 1))
            vector_C5F1 = np.zeros((self.C5F1.shape[0], 1))
            vector_C7F1 = np.zeros((self.C7F1.shape[0], 1))
            vector_CMF2 = np.zeros((self.CMF2.shape[0], 1))
            for d in range(self.C1F1.shape[0]):
                for a in range(1):
                    vector_C1F1[d][a] = random.gauss(0, 0.1)
            for d in range(self.C3F1.shape[0]):
                for a in range(1):
                    vector_C3F1[d][a] = random.gauss(0, 0.1)
            for d in range(self.C5F1.shape[0]):
                for a in range(1):
                    vector_C5F1[d][a] = random.gauss(0, 0.1)
            for d in range(self.C7F1.shape[0]):
                for a in range(1):
                    vector_C7F1[d][a] = random.gauss(0, 0.1)
            for d in range(self.CMF2.shape[0]):
                for a in range(1):
                    vector_CMF2[d][a] = random.gauss(0, 0.1)
            self.C1F1 = add_at_pos_each_row(self.C1F1, vector_C1F1, to_add)
            self.C3F1 = add_at_pos_each_row(self.C3F1, vector_C3F1, to_add)
            self.C5F1 = add_at_pos_each_row(self.C5F1, vector_C5F1, to_add)
            self.C7F1 = add_at_pos_each_row(self.C7F1, vector_C7F1, to_add)
            self.CMF2 = add_at_pos_each_row(self.CMF2, vector_CMF2, to_add)

        # add filters (part 1)
        if random.random() < - 1:
            self.previous_mutation_vector = []
            target = random.randint(0, 2)
            if target == 0:
                new_filter = np.zeros((1, self.C3F1.shape[1]))
                for w in range(self.C3F1.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C3F1 = np.concatenate((self.C3F1, new_filter))
                to_add = 9
                vector_C3F2 = np.zeros((self.C3F2.shape[0], to_add))
                for d in range(self.C3F2.shape[0]):
                    for a in range(to_add):
                        vector_C3F2[d][a] = random.gauss(0, 0.1)
                self.C3F2 = np.concatenate((self.C3F2, vector_C3F2), axis=1)
            elif target == 1:
                new_filter = np.zeros((1, self.C5F1.shape[1]))
                for w in range(self.C5F1.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C5F1 = np.concatenate((self.C5F1, new_filter))
                to_add = 9
                vector_C5F2 = np.zeros((self.C5F2.shape[0], to_add))
                for d in range(self.C5F2.shape[0]):
                    for a in range(to_add):
                        vector_C5F2[d][a] = random.gauss(0, 0.1)
                self.C5F2 = np.concatenate((self.C5F2, vector_C5F2), axis=1)
            else:
                new_filter = np.zeros((1, self.C7F1.shape[1]))
                for w in range(self.C7F1.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C7F1 = np.concatenate((self.C7F1, new_filter))
                to_add = 9
                vector_C7F2 = np.zeros((self.C7F2.shape[0], to_add))
                for d in range(self.C7F2.shape[0]):
                    for a in range(to_add):
                        vector_C7F2[d][a] = random.gauss(0, 0.1)
                self.C7F2 = np.concatenate((self.C7F2, vector_C7F2), axis=1)

        # add filters (part 2)
        if random.random() < -1:
            self.previous_mutation_vector = []
            target = random.randint(0, 4)
            if target == 0:
                new_filter = np.zeros((1, self.C1F1.shape[1]))
                for w in range(self.C3F1.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C1F1 = np.concatenate((self.C1F1, new_filter))
                return self.C1F1.shape[0]
            elif target == 1:
                new_filter = np.zeros((1, self.C3F2.shape[1]))
                for w in range(self.C3F2.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C3F2 = np.concatenate((self.C3F2, new_filter))
                return self.C1F1.shape[0] + self.C3F2.shape[0]
            elif target == 2:
                new_filter = np.zeros((1, self.C5F2.shape[1]))
                for w in range(self.C5F2.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C5F2 = np.concatenate((self.C5F2, new_filter))
                return self.C1F1.shape[0] + self.C3F2.shape[0] + self.C5F2.shape[0]
            elif target == 3:
                new_filter = np.zeros((1, self.C7F2.shape[1]))
                for w in range(self.C7F2.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.C7F2 = np.concatenate((self.C7F2, new_filter))
                return self.C1F1.shape[0] + self.C3F2.shape[0] + self.C5F2.shape[0] + self.C7F2.shape[0]
            else:
                new_filter = np.zeros((1, self.CMF2.shape[1]))
                for w in range(self.CMF2.shape[1]):
                    new_filter[0][w] = random.gauss(0, 1)
                self.CMF2 = np.concatenate((self.CMF2, new_filter))
                return self.C1F1.shape[0] + self.C3F2.shape[0] + self.C5F2.shape[0] + self.CMF2.shape[0]
        return None

    def decay(self):
        for i in range(len(self.previous_mutation_vector)):
            self.previous_mutation_vector[i] *= 0.9

    def run(self, channels):
        d = cuda.to_device(channels)
        height = channels.shape[1]
        width = channels.shape[2]
        C1F1Outs = np.zeros((self.C1F1.shape[0], height, width))
        d1 = cuda.to_device(C1F1Outs)
        C3F1Outs = np.zeros((self.C3F1.shape[0], height, width))
        d2 = cuda.to_device(C3F1Outs)
        C5F1Outs = np.zeros((self.C5F1.shape[0], height, width))
        d3 = cuda.to_device(C5F1Outs)
        C7F1Outs = np.zeros((self.C7F1.shape[0], height, width))
        d4 = cuda.to_device(C7F1Outs)
        CMF1Outs = np.zeros((channels.shape[0], height, width))
        d5 = cuda.to_device(CMF1Outs)
        threadsperblock = 64
        blockspergrid = (C1F1Outs.size + C3F1Outs.size + C5F1Outs.size + C7F1Outs.size + CMF1Outs.size + threadsperblock - 1) // threadsperblock
        inception_gpu_1[blockspergrid, threadsperblock](d, self.C1F1, self.C3F1, self.C5F1, self.C7F1,
                                                        d1, d2, d3, d4, d5)
        C3F2Outs = np.zeros((self.C3F2.shape[0], height, width))
        C5F2Outs = np.zeros((self.C5F2.shape[0], height, width))
        C7F2Outs = np.zeros((self.C7F2.shape[0], height, width))
        CMF2Outs = np.zeros((self.CMF2.shape[0], height, width))
        blockspergrid = (
                                    C3F2Outs.size + C5F2Outs.size + C7F2Outs.size + CMF2Outs.size + threadsperblock - 1) // threadsperblock
        inception_gpu_2[blockspergrid, threadsperblock](d2, d3, d4, d5,
                                                        self.C3F2, self.C5F2, self.C7F2, self.CMF2,
                                                        C3F2Outs, C5F2Outs, C7F2Outs, CMF2Outs)

        C1F1Outs = d1.copy_to_host()
        outs = np.zeros((C1F1Outs.shape[0] + C3F2Outs.shape[0] + C5F2Outs.shape[0] + C7F2Outs.shape[0] +
                         CMF2Outs.shape[0], height, width))
        outs[0:C1F1Outs.shape[0]] = C1F1Outs
        outs[C1F1Outs.shape[0]:C1F1Outs.shape[0] + C3F2Outs.shape[0]] = C3F2Outs
        outs[
        C1F1Outs.shape[0] + C3F2Outs.shape[0]: C1F1Outs.shape[0] + C3F2Outs.shape[0] + C5F2Outs.shape[0]] = C5F2Outs
        outs[C1F1Outs.shape[0] + C3F2Outs.shape[0] + C5F2Outs.shape[0]:
             C1F1Outs.shape[0] + C3F2Outs.shape[0] + C5F2Outs.shape[0] + C7F2Outs.shape[0]] = C7F2Outs
        outs[C1F1Outs.shape[0] + C3F2Outs.shape[0] + C5F2Outs.shape[0] + C7F2Outs.shape[0]:
             C1F1Outs.shape[0] + C3F2Outs.shape[0] + C5F2Outs.shape[0] + C7F2Outs.shape[0] + CMF2Outs.shape[
                 0]] = CMF2Outs
        return outs

    def __repr__(self):
        string = "{"
        string += '"C1F1": ' + repr(self.C1F1.tolist()) + ", "
        string += '"C3F1": ' + repr(self.C3F1.tolist()) + ", "
        string += '"C5F1": ' + repr(self.C5F1.tolist()) + ", "
        string += '"C7F1": ' + repr(self.C7F1.tolist()) + ", "
        string += '"C3F2": ' + repr(self.C3F2.tolist()) + ", "
        string += '"C5F2": ' + repr(self.C5F2.tolist()) + ", "
        string += '"C7F2": ' + repr(self.C7F2.tolist()) + ", "
        string += '"CMF2": ' + repr(self.CMF2.tolist()) + ", "
        string += '"max_pool" : false'
        return string + "}"


@cuda.jit
def inception_gpu_2(C3F1O, C5F1O, C7F1O, CMF1O, C3F2, C5F2, C7F2, CMF2, C3F2O, C5F2O, C7F2O, CMF2O):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    pos = tx + ty * bw

    LOCAL_VAR = float(0)

    if pos < C3F2O.size:
        channel_target = pos // (C3F1O.shape[1] * C3F1O.shape[2])
        pos_inner = pos % (C3F1O.shape[1] * C3F1O.shape[2])
        y_coord = pos_inner // C3F1O.shape[1]
        x_coord = pos_inner % C3F1O.shape[1]
        LOCAL_VAR += C3F2[channel_target][0]
        for c in range(C3F1O.shape[0]):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    if 0 <= y_coord + y < C3F1O.shape[1] and 0 <= x_coord + x < C3F1O.shape[2] \
                            and 1 + 9 * c + 3 * (y + 1) + x + 1 < C3F2.shape[1]:
                        LOCAL_VAR += C3F1O[c][y_coord + y][x_coord + x] \
                                     * C3F2[channel_target][1 + 9 * c + 3 * (y + 1) + x + 1]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C3F2O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C3F2O.size + C5F2O.size:
        pos = pos - C3F2O.size
        channel_target = pos // (C5F1O.shape[1] * C5F1O.shape[2])
        pos_inner = pos % (C5F1O.shape[1] * C5F1O.shape[2])
        y_coord = pos_inner // C5F1O.shape[1]
        x_coord = pos_inner % C5F1O.shape[1]
        LOCAL_VAR += C5F2[channel_target][0]
        for c in range(C5F1O.shape[0]):
            for y in range(-2, 3):
                for x in range(-2, 3):
                    if 0 <= y_coord + y < C5F1O.shape[1] and 0 <= x_coord + x < C5F1O.shape[2] \
                            and 1 + 25 * c + 5 * (y + 1) + x + 1 < C5F2.shape[1]:
                        LOCAL_VAR += C5F1O[c][y_coord + y][x_coord + x] \
                                     * C5F2[channel_target][1 + 25 * c + 5 * (y + 1) + x + 1]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C5F2O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C3F2O.size + C5F2O.size + C7F2O.size:
        pos = pos - C3F2O.size - C5F2O.size
        channel_target = pos // (C7F1O.shape[1] * C7F1O.shape[2])
        pos_inner = pos % (C7F1O.shape[1] * C7F1O.shape[2])
        y_coord = pos_inner // C7F1O.shape[1]
        x_coord = pos_inner % C7F1O.shape[1]
        LOCAL_VAR += C7F2[channel_target][0]
        for c in range(C7F1O.shape[0]):
            for y in range(-3, 4):
                for x in range(-3, 4):
                    if 0 <= y_coord + y < C7F1O.shape[1] and 0 <= x_coord + x < C7F1O.shape[2] \
                            and 1 + 49 * c + 7 * (y + 1) + x + 1 < C7F2.shape[1]:
                        LOCAL_VAR += C7F1O[c][y_coord + y][x_coord + x] \
                                     * C7F2[channel_target][1 + 49 * c + 7 * (y + 1) + x + 1]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C7F2O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C3F2O.size + C5F2O.size + C7F2O.size + CMF2O.size:
        pos = pos - C3F2O.size - C5F2O.size - C7F2O.size
        channel_target = pos // (CMF1O.shape[1] * CMF1O.shape[2])
        pos_inner = pos % (CMF1O.shape[1] * CMF1O.shape[2])
        y_coord = pos_inner // CMF1O.shape[1]
        x_coord = pos_inner % CMF1O.shape[1]
        if channel_target >= CMF2.shape[0] or channel_target >= CMF2.shape[0]:
            return
        if y_coord >= CMF1O.shape[1] or x_coord >= CMF1O.shape[2]:
            return
        LOCAL_VAR += CMF2[channel_target][0]
        for c in range(1, CMF1O.shape[0] + 1):
            LOCAL_VAR += CMF1O[c - 1][y_coord][x_coord] * CMF2[channel_target][c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        CMF2O[channel_target][y_coord][x_coord] = LOCAL_VAR


@cuda.jit
def inception_gpu_1_multiple(images, C1F1, C3F1, C5F1, C7F1, C1F1O, C3F1O, C5F1O, C7F1O, CMF1O):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    pos_outer = tx + ty * bw

    LOCAL_VAR = float(0)

    target = pos_outer // (C1F1O[0].size + C3F1O[0].size + C5F1O[0].size + C7F1O[0].size + CMF1O[0].size)

    pos = pos_outer % (C1F1O[0].size + C3F1O[0].size + C5F1O[0].size + C7F1O[0].size + CMF1O[0].size)

    if pos < C1F1O[target].size:
        channel_target = pos // (images[target].shape[1] * images[target].shape[2])
        pos_inner = pos % (images[target].shape[1] * images[target].shape[2])
        y_coord = pos_inner // images[target].shape[1]
        x_coord = pos_inner % images[target].shape[1]
        if channel_target >= C1F1.shape[0] or channel_target >= C1F1O[target].shape[0]:
            return
        if y_coord >= images[target].shape[1] or x_coord >= images[target].shape[2]:
            return
        LOCAL_VAR += C1F1[channel_target][0]
        for c in range(1, images[target].shape[0] + 1):
            LOCAL_VAR += images[target][c - 1][y_coord][x_coord] * C1F1[channel_target][c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C1F1O[target][channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O[target].size + C3F1O[target].size:
        pos = pos - C1F1O[target].size
        channel_target = pos // (images[target].shape[1] * images[target].shape[2])
        pos_inner = pos % (images[target].shape[1] * images[target].shape[2])
        y_coord = pos_inner // images[target].shape[1]
        x_coord = pos_inner % images[target].shape[1]
        if channel_target >= C3F1.shape[0] or channel_target >= C3F1O[target].shape[0]:
            return
        if y_coord >= images[target].shape[1] or x_coord >= images[target].shape[2]:
            return
        LOCAL_VAR += C3F1[channel_target][0]
        for c in range(images[target].shape[0]):
            LOCAL_VAR += images[target][c - 1][y_coord][x_coord] * C3F1[channel_target][1 + c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C3F1O[target][channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O[target].size + C3F1O[target].size + C5F1O[target].size:
        pos = pos - C1F1O[target].size - C3F1O[target].size
        channel_target = pos // (images[target].shape[1] * images[target].shape[2])
        pos_inner = pos % (images[target].shape[1] * images[target].shape[2])
        y_coord = pos_inner // images[target].shape[1]
        x_coord = pos_inner % images[target].shape[1]
        if channel_target >= C5F1.shape[0] or channel_target >= C5F1O[target].shape[0]:
            return
        if y_coord >= images[target].shape[1] or x_coord >= images[target].shape[2]:
            return
        LOCAL_VAR += C5F1[channel_target][0]
        for c in range(images[target].shape[0]):
            LOCAL_VAR += images[target][c][y_coord][x_coord] * C5F1[channel_target][1 + c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C5F1O[target][channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O[target].size + C3F1O[target].size + C5F1O[target].size + C7F1O[target].size:
        pos = pos - C1F1O[target].size - C3F1O[target].size - C5F1O[target].size
        channel_target = pos // (images[target].shape[1] * images[target].shape[2])
        pos_inner = pos % (images[target].shape[1] * images[target].shape[2])
        y_coord = pos_inner // images[target].shape[1]
        x_coord = pos_inner % images[target].shape[1]
        if channel_target >= C7F1.shape[0] or channel_target > C7F1O[target].shape[0]:
            return
        if y_coord >= images[target].shape[1] or x_coord >= images[target].shape[2]:
            return
        LOCAL_VAR += C7F1[channel_target][0]
        for c in range(images[target].shape[0]):
            LOCAL_VAR += images[target][c][y_coord][x_coord] * C7F1[channel_target][1 + c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C7F1O[target][channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O[target].size + C3F1O[target].size + C5F1O[target].size + C7F1O[target].size + CMF1O[target].size:
        pos = pos - C1F1O[target].size - C3F1O[target].size - C5F1O[target].size
        channel_target = pos // (images[target].shape[1] * images[target].shape[2])
        pos_inner = pos % (images[target].shape[1] * images[target].shape[2])
        y_coord = pos_inner // images[target].shape[1]
        x_coord = pos_inner % images[target].shape[1]
        if channel_target >= CMF1O[target].shape[0]:
            return
        if y_coord >= images[target].shape[1] or x_coord >= images[target].shape[2]:
            return
        LOCAL_VAR = -100
        for c in range(images[target].shape[0]):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    if 0 <= y_coord + y < images[target].shape[1] and 0 <= x_coord + x < images[target].shape[2]:
                        if LOCAL_VAR < images[target][c][y_coord + y][x_coord + x]:
                            LOCAL_VAR = images[target][c][y_coord + y][x_coord + x]
        CMF1O[target][channel_target][y_coord][x_coord] = LOCAL_VAR

@cuda.jit
def inception_gpu_1(channels, C1F1, C3F1, C5F1, C7F1, C1F1O, C3F1O, C5F1O, C7F1O, CMF1O):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    pos = tx + ty * bw

    LOCAL_VAR = float(0)

    if pos < C1F1O.size:
        channel_target = pos // (channels.shape[1] * channels.shape[2])
        pos_inner = pos % (channels.shape[1] * channels.shape[2])
        y_coord = pos_inner // channels.shape[1]
        x_coord = pos_inner % channels.shape[1]
        if channel_target >= C1F1.shape[0] or channel_target >= C1F1O.shape[0]:
            return
        if y_coord >= channels.shape[1] or x_coord >= channels.shape[2]:
            return
        LOCAL_VAR += C1F1[channel_target][0]
        for c in range(1, channels.shape[0] + 1):
            LOCAL_VAR += channels[c - 1][y_coord][x_coord] * C1F1[channel_target][c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C1F1O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O.size + C3F1O.size:
        pos = pos - C1F1O.size
        channel_target = pos // (channels.shape[1] * channels.shape[2])
        pos_inner = pos % (channels.shape[1] * channels.shape[2])
        y_coord = pos_inner // channels.shape[1]
        x_coord = pos_inner % channels.shape[1]
        if channel_target >= C3F1.shape[0] or channel_target >= C3F1O.shape[0]:
            return
        if y_coord >= channels.shape[1] or x_coord >= channels.shape[2]:
            return
        LOCAL_VAR += C3F1[channel_target][0]
        for c in range(channels.shape[0]):
            LOCAL_VAR += channels[c - 1][y_coord][x_coord] * C3F1[channel_target][1 + c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C3F1O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O.size + C3F1O.size + C5F1O.size:
        pos = pos - C1F1O.size - C3F1O.size
        channel_target = pos // (channels.shape[1] * channels.shape[2])
        pos_inner = pos % (channels.shape[1] * channels.shape[2])
        y_coord = pos_inner // channels.shape[1]
        x_coord = pos_inner % channels.shape[1]
        if channel_target >= C5F1.shape[0] or channel_target >= C5F1O.shape[0]:
            return
        if y_coord >= channels.shape[1] or x_coord >= channels.shape[2]:
            return
        LOCAL_VAR += C5F1[channel_target][0]
        for c in range(channels.shape[0]):
            LOCAL_VAR += channels[c][y_coord][x_coord] * C5F1[channel_target][1 + c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C5F1O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O.size + C3F1O.size + C5F1O.size + C7F1O.size:
        pos = pos - C1F1O.size - C3F1O.size - C5F1O.size
        channel_target = pos // (channels.shape[1] * channels.shape[2])
        pos_inner = pos % (channels.shape[1] * channels.shape[2])
        y_coord = pos_inner // channels.shape[1]
        x_coord = pos_inner % channels.shape[1]
        if channel_target >= C7F1.shape[0] or channel_target > C7F1O.shape[0]:
            return
        if y_coord >= channels.shape[1] or x_coord >= channels.shape[2]:
            return
        LOCAL_VAR += C7F1[channel_target][0]
        for c in range(channels.shape[0]):
            LOCAL_VAR += channels[c][y_coord][x_coord] * C7F1[channel_target][1 + c]
        if LOCAL_VAR < 0:
            LOCAL_VAR = 0
        C7F1O[channel_target][y_coord][x_coord] = LOCAL_VAR

    elif pos < C1F1O.size + C3F1O.size + C5F1O.size + C7F1O.size + CMF1O.size:
        pos = pos - C1F1O.size - C3F1O.size - C5F1O.size
        channel_target = pos // (channels.shape[1] * channels.shape[2])
        pos_inner = pos % (channels.shape[1] * channels.shape[2])
        y_coord = pos_inner // channels.shape[1]
        x_coord = pos_inner % channels.shape[1]
        if channel_target >= CMF1O.shape[0]:
            return
        if y_coord >= channels.shape[1] or x_coord >= channels.shape[2]:
            return
        LOCAL_VAR = -100
        for c in range(channels.shape[0]):
            for y in range(-1, 2):
                for x in range(-1, 2):
                    if 0 <= y_coord + y < channels.shape[1] and 0 <= x_coord + x < channels.shape[2]:
                        if LOCAL_VAR < channels[c][y_coord + y][x_coord + x]:
                            LOCAL_VAR = channels[c][y_coord + y][x_coord + x]
        CMF1O[channel_target][y_coord][x_coord] = LOCAL_VAR


class conv_network:
    def __init__(self, width, height, depth, out_count):
        self.width = width
        self.height = height
        self.depth = depth
        width_tmp = width
        height_tmp = height
        self.out_count = out_count
        self.mutation_count = 0

        depth_inner = max(out_count, 64)
        self.big_network = [Inception(width_tmp, height_tmp, depth, depth_inner)]
        while width_tmp + height_tmp > 4:
            self.big_network.append(Max_Pool_Section(False))
            width_tmp = int((width_tmp - 1) / 2)
            height_tmp = int((height_tmp - 1) / 2)
            self.big_network.append(Inception(width_tmp, height_tmp, depth_inner, depth_inner))
            depth_inner += 16
        self.final_layer = Inception(1, 1, depth_inner, out_count, final_layer=True)

    def mutate_inceptions(self):
        self.mutation_count += 1
        new_channels = 0
        for inception_id in range(0, len(self.big_network), 2):
            new_channels = self.big_network[inception_id].mutate(new_channels)
        self.final_layer.mutate(new_channels)

    def decay(self):
        for inception_id in range(0, len(self.big_network), 2):
            new_channels = self.big_network[inception_id].decay()

    def run(self, image, show_as_images=False, multiple_images=False):
        channels = image
        i = 0
        if show_as_images and not multiple_images:
            self.show_images(channels, self.depth)
        for layer in self.big_network:
            i += 1
            channels = layer.run(channels)
            if not multiple_images and show_as_images and type(layer) == Inception or str(
                    type(layer)) == "<class 'conv_network_new.Section'>" or str(
                    type(layer)) == "<class '__main__.Section'>":
                self.show_images(channels, layer.depth_out)

        outs = self.final_layer.run(channels)[:self.out_count]
        outs = outs.tolist()
        to_return = []
        for o in outs:
            biggest = None
            for y in o:
                for x in y:
                    if biggest is None or x > biggest:
                        biggest = x
            to_return.append(biggest)
        if min(to_return) < 0:
            tmp = min(to_return)
            for i in range(self.out_count):
                to_return[i] += tmp
        if sum(to_return) > 0:
            tmp = sum(to_return)
            for i in range(self.out_count):
                to_return[i] /= tmp
        if show_as_images:
            always_same = True
            for i in range(100):
                if self.run(image) != to_return:
                    always_same = False
                    break
            print("Always same: " + str(always_same))
        return to_return

    def show_images(self, panes, pane_count):
        # combine images together to make one big image
        width = math.sqrt(pane_count) // 1
        if math.sqrt(pane_count) % 1 > 0.5:
            width += 1
        height = pane_count // width
        if pane_count % width > 0:
            height += 1
        width = int(width)
        height = int(height)
        current_pane = 0
        image = np.zeros(((panes.shape[1] + 1) * height, (panes.shape[2] + 1) * width))
        flag = False
        for y in range(height):
            for x in range(width):
                for y1 in range(panes.shape[1]):
                    for x1 in range(panes.shape[2]):
                        image[y * (panes.shape[1] + 1) + y1][x * (panes.shape[2] + 1) + x1] = panes[current_pane][y1][
                            x1]
                current_pane += 1
                if current_pane >= pane_count:
                    flag = True
                    break
            if flag:
                break
        plt.imshow(image, cmap="gray")
        plt.show()

    def __repr__(self):
        string = "{"
        string += '"width": ' + repr(self.width) + ', '
        string += '"height": ' + repr(self.height) + ", "
        string += '"depth": ' + repr(self.depth) + ", "
        string += '"out_count": ' + repr(self.out_count) + ", "
        string += '"big_network": ' + repr(self.big_network) + ", "
        string += '"final_layer": ' + repr(self.final_layer) + "}"
        return string


def mutate_individual(conv_network):
    conv_network.mutate_inceptions()
    return conv_network


if __name__ == "__main__":
    import json

    conv = conv_network(105, 105, 10, 10)
    print(repr(conv))
