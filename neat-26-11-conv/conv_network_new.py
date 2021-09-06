import math
import time

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
    def __init__(self):
        pass

    def run(self, panes):
        outputs_new = np.zeros((panes.shape[0], panes.shape[1], panes.shape[2] // 3 + 1, panes.shape[3] // 3 + 1))
        threadsperblock = 64
        blockspergrid = (panes.shape[0] * panes.shape[1] * (panes.shape[2] // 3 + 1) * (panes.shape[3] // 3 + 1) + (
                    threadsperblock - 1)) // threadsperblock
        max_pool_gpu[blockspergrid, threadsperblock](panes, outputs_new)
        return outputs_new

    def __repr__(self):
        return "{max_pool: True}"


# outputs = (pane_count, new_width, new_height)
# panes = (pane_count, width, height
@cuda.jit
def max_pool_gpu(images, outputs): #image, pane, y, x
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    pos1 = tx + ty * bw
    target_image = pos1 // outputs[0].size
    
    pos2 = pos1 % outputs[0].size
    target_pane = pos2 // outputs[0][0].size
    
    pos3 = pos2 % outputs[0][0].size
    y = (pos3 // outputs.shape[3]) * 3
    x = (pos3 % outputs.shape[3]) * 3

    if target_image >= images.shape[0] or target_pane >= images.shape[1] or y >= images.shape[2] or x >= images.shape[3]:
        return
    max = images[target_image][target_pane][y-1][x-1]
    if images[target_image][target_pane][y-1][x] > max:
        max = images[target_image][target_pane][y-1][x]
    if x < len(images[target_image][target_pane][0]) - 1 and images[target_image][target_pane][y-1][x+1] > max:
        max = images[target_image][target_pane][y-1][x+1]

    if images[target_image][target_pane][y][x-1] > max:
        max = images[target_image][target_pane][y][x-1]

    if images[target_image][target_pane][y][x] > max:
        max = images[target_image][target_pane][y][x]
    if x < len(images[target_image][target_pane][0]) - 1 and images[target_image][target_pane][y][x+1] > max:
        max = images[target_image][target_pane][y][x+1]

    if y < len(images[target_image][target_pane])-1 and images[target_image][target_pane][y+1][x-1] > max:
        max = images[target_image][target_pane][y+1][x-1]
    if y < len(images[target_image][target_pane])-1 and images[target_image][target_pane][y+1][x] > max:
        max = images[target_image][target_pane][y+1][x]
    if y < len(images[target_image][target_pane])-1 and x < len(images[target_image][target_pane][0]) - 1 and images[target_image][target_pane][y+1][x+1] > max:
        max = images[target_image][target_pane][y+1][x+1]
    outputs[target_image][target_pane][y // 3][x // 3] = max



@cuda.jit
def pane_to_input(panes, tmp):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    posBig = tx + ty * bw
    target_image = posBig // panes[0].size

    pos = posBig % panes[0].size
    p_target = pos // (panes.shape[2] * panes.shape[3])
    inner_pos = pos % (panes.shape[2] * panes.shape[3])
    y = inner_pos // panes.shape[3]
    x = inner_pos % panes.shape[3]
    tmp_pos = panes.shape[2] * y + x
    if target_image >= panes.shape[0] or p_target >= panes.shape[1] or y >= panes.shape[2] or x >= panes.shape[3]:
        return
    tmp[target_image][tmp_pos][0] = 1
    if y > 0 and x > 0:
        tmp[target_image][tmp_pos][1 + p_target*9] = panes[target_image][p_target][y-1][x-1]
    if y > 0:
        tmp[target_image][tmp_pos][2 + p_target*9] = panes[target_image][p_target][y-1][x]
    if y > 0 and x < len(panes[target_image][p_target][0]) - 1:
        tmp[target_image][tmp_pos][3 + p_target*9] = panes[target_image][p_target][y-1][x+1]

    if x > 0:
        tmp[target_image][tmp_pos][4 + p_target*9] = panes[target_image][p_target][y][x-1]
    tmp[target_image][tmp_pos][5 + p_target*9] = panes[target_image][p_target][y][x]
    if x < len(panes[target_image][p_target][0]) - 1:
        tmp[target_image][tmp_pos][6 + p_target*9] = panes[target_image][p_target][y][x+1]

    if y < len(panes[target_image][p_target])-1 and x > 0:
        tmp[target_image][tmp_pos][7 + p_target*9] = panes[target_image][p_target][y+1][x-1]
    if y < len(panes[target_image][p_target])-1:
        tmp[target_image][tmp_pos][8 + p_target*9] = panes[target_image][p_target][y+1][x]
    if y < len(panes[target_image][p_target])-1 and x < len(panes[target_image][p_target][0]) - 1:
        tmp[target_image][tmp_pos][9 + p_target*9] = panes[target_image][p_target][y+1][x+1]


class Filter:
    def __init__(self, network, frame_size):
        self.network = network
        self.frame_size = frame_size

    def get_inputs(self, panes):
        threadsperblock = 64
        blockspergrid = (panes.size + (threadsperblock - 1)) // threadsperblock
        tmp = np.zeros((panes.shape[0], panes.shape[2] * panes.shape[3], panes.shape[1] * self.frame_size**2 + 1), dtype=float)

        pane_to_input[blockspergrid, threadsperblock](panes, tmp)
        return tmp

    def run(self, panes):
        threadsperblock = 64

        inputs = self.get_inputs(panes)

        blockspergrid = (inputs.size + (threadsperblock - 1)) // threadsperblock

        outputs = np.zeros((panes.shape[0], panes.shape[2] * panes.shape[3]), dtype=float)
        node_vals = np.zeros((inputs.shape[0], inputs.shape[1], len(self.network.nodes)), dtype=float)

        if type(self.network) == ArtificialNetwork.Network:
            commands, connection_weights, connection_targets, sources, input_ids = \
                self.network.get_network_gpu_representation()

            ArtificialNetwork.run_network_gpu[blockspergrid, threadsperblock](inputs, commands, connection_weights,
                                          connection_targets, sources, commands.size, self.network.original_in_count,
                                          self.network.in_count, input_ids, outputs, node_vals)
        else:
            CGP.run_network_gpu[blockspergrid, threadsperblock](inputs, self.network.nodes, self.network.in_count, self.network.out_count, self.network.node_count, outputs, node_vals)
            outputs /= 10

        new_outputs = np.reshape(outputs, (panes.shape[0], *panes.shape[2:]))
        return new_outputs

    def __repr__(self):
        string = "{"
        string += "network: " + repr(self.network) + ", "
        string += "frame_size: " + repr(self.frame_size) + "}"
        return string



class Section:
    def __init__(self, width, height, pane_count):
        self.width = width
        self.height = height
        self.pane_count_in = pane_count
        self.pane_count_out = pane_count
        self.filters = []

    def combine_panes(self, panes, paneNo):
        if paneNo <= 1:
            return panes[0]
        output = []
        for y in range(len(panes[0])):
            output.append([])
            for x in range(len(panes[0])):
                output[-1].append(panes[0][y][x] / paneNo)
                for pane in range(1, paneNo):
                    output[-1][-1] += panes[pane][y][x] / paneNo
        return output

    def run(self, panes):
        for filter in self.filters:
            inputs = np.zeros((panes.shape[0], len(filter[1]), *panes.shape[2:]))
            for t in range(panes.shape[0]):
                for p in range(len(filter[1])):
                    inputs[t][p] = panes[t][p]
            tmp = filter[0].run(inputs)
            for t in range(tmp.shape[0]):
                if type(filter[2]) == int:
                    panes[t][filter[2]] = tmp[t]
                else:
                    for i in range(len(filter[2])):
                        panes[t][filter[2][i]] = tmp[t][i]
        return panes

    def mut_add_filter(self):
        applicable = 0
        for f in self.filters:
            if f[2] is None:
                applicable += 1
        inputs_to_filter = [random.randint(0, applicable + self.pane_count_in-1)]
        willReplace = random.choice([True, False])
        if willReplace:
            replace = inputs_to_filter[0]
        else:
            replace = self.pane_count_out
            self.pane_count_out += 1
        if random.random() < 0.7:
            self.filters.append([Filter(ArtificialNetwork.Network(10, 1, False, output_activation="relu"), 3), inputs_to_filter, replace])
        else:
            self.filters.append([Filter(CGP.CGP(9, 1, 100), 3), inputs_to_filter, replace])
        return willReplace

    def mut_add_glcm(self):
        applicable = 0
        for f in self.filters:
            if f[2] is None:
                applicable += 1
        inputs_to_filter = [random.randint(0, applicable + self.pane_count_in-1)]
        outputs = np.array(range(self.pane_count_out, self.pane_count_out + 4))
        self.pane_count_out += 4
        self.filters.append([GLCM.GLCM_Filter(np.array([random.randint(-2, 2), random.randint(-2, 2)])), inputs_to_filter, outputs])

    def mut_add_connection(self):
        if len(self.filters) == 0:
            return
        f_choice = random.choice(list(range(len(self.filters))))
        if type(self.filters[f_choice][0]) == GLCM.GLCM_Filter: #TODO: make glcm filter accept more inputs
            return
        current_cons = self.filters[f_choice][1]
        potential_new_cons = []
        for con in range(f_choice):
            if con not in current_cons:
                potential_new_cons.append(con)
        if len(potential_new_cons) > 0:
            self.filters[f_choice][1].append(random.choice(potential_new_cons))
            self.filters[f_choice][0].network.add_inputs(9)

    def mut_random_filter(self):
        if len(self.filters) == 0:
            return
        choice = random.choice(self.filters)
        if type(choice[0]) == Filter:
            if type(choice[0].network) == ArtificialNetwork.Network:
                ArtificialNetwork.mutate_individual(choice[0].network)
            else:
                CGP.mutate_individual(choice[0].network)
        if type(choice[0]) == GLCM.GLCM_Filter:
            GLCM.mutate_individual(choice[0])


    def add_pane(self):
        for filter in self.filters:
            filter[1] = list(map(lambda x: x+1, filter[1]))
            if filter[2] is not None:
                filter[2] += 1
        self.pane_count_in += 1
        self.pane_count_out += 1

    def __repr__(self):
        string = "{ width: " + repr(self.width) + ", "
        string += "height: " + repr(self.height) + ", "
        string += "pane_count_in: " + repr(self.pane_count_in) + ", "
        string += "pane_count_out: " + repr(self.pane_count_out) + ", "
        string += "max_pool: " + repr(False) + ", "
        string += "filters: " + repr(self.filters) + "}"
        return string


class conv_network:
    def __init__(self, width, height, depth, out_count, section_count, one_shot = False, diffusion_gradients = 0):
        self.width = width
        self.height = height
        self.depth = depth
        width_tmp = width
        height_tmp = height

        if one_shot and diffusion_gradients % out_count > 0:
            raise ValueError("The number of diffusion gradients must be divisible by the number of outputs")

        self.section_count = section_count
        self.big_network = []
        for i in range(section_count):
            self.big_network.append(Section(width_tmp, height_tmp, depth))
            self.big_network.append(Max_Pool_Section())
            width_tmp = (width_tmp + 1) // 3
            height_tmp = (height_tmp + 1)//3
        self.final_inputs_per_filter = width_tmp * height_tmp
        self.final_layer = ArtificialNetwork.Network(self.final_inputs_per_filter * depth + 1, out_count, False, output_activation="relu", diffusion_gradients=diffusion_gradients)
        self.second_final_layer = None
        if one_shot:
            self.second_final_layer = ArtificialNetwork.Network(self.final_inputs_per_filter * depth + 1, diffusion_gradients // out_count, False, output_activation="sigmoid")

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
                        image[y*(panes.shape[1] + 1) + y1][x*(panes.shape[2] + 1) + x1] = panes[current_pane][y1][x1]
                current_pane += 1
                if current_pane >= pane_count:
                    flag = True
                    break
            if flag:
                break
        plt.imshow(image, cmap="gray")
        plt.show()

    def run(self, images, show_as_images = False, example = False):
        panes = np.zeros((images.shape[0], self.get_total_panes(), *images.shape[2:]))
        for im in range(images.shape[0]):
            for p in range(images.shape[1]):
                panes[im][p] = images[im][p]
        i = 0
        for layer in self.big_network:
            i += 1
            panes = layer.run(panes)
            if show_as_images and str(type(layer)) == "<class 'conv_network_new.Section'>" or str(type(layer)) == "<class '__main__.Section'>":
                self.show_images(panes[0], layer.pane_count_out)
        outputs_big = np.zeros((len(images), self.final_layer.out_count))
        i = 0
        for pane in panes:
            flat_inputs = []
            for p in pane:
                for y in p:
                    flat_inputs.extend(y)
            if example:
                outputs = np.array(self.second_final_layer.run([1] + flat_inputs))
                rewards = np.zeros((self.final_layer.diffusion_gradients))
                rewards[self.second_final_layer.out_count * i: self.second_final_layer.out_count * (i+1)] = outputs
                self.final_layer.learn(rewards)
            else:
                outputs = np.array(self.final_layer.run([1] + flat_inputs))
                if sum(outputs) != 0:
                    outputs /= sum(outputs)
                outputs_big[i] = outputs
            i += 1
        return outputs_big

    def mut_add_filter(self):
        sections = list(range(0, len(self.big_network), 2))
        choice = random.choice(sections)
        if not self.big_network[choice].mut_add_filter():
            for i in range(choice+2, len(self.big_network), 2):
                self.big_network[i].add_pane()
            self.final_layer.add_inputs(self.final_inputs_per_filter)
            if self.second_final_layer is not None:
                self.second_final_layer.add_inputs(self.final_inputs_per_filter)

    def mut_add_glcm(self):
        sections = list(range(0, len(self.big_network), 2))
        choice = random.choice(sections)
        self.big_network[choice].mut_add_glcm()
        for i in range(choice+2, len(self.big_network), 2):
            for j in range(4):
                self.big_network[i].add_pane()
        self.final_layer.add_inputs(4*self.final_inputs_per_filter)
        if self.second_final_layer is not None:
            self.second_final_layer.add_inputs(4*self.final_inputs_per_filter)

    def mut_add_connection(self):
        section_choice = random.choice(list(range(0, len(self.big_network), 2)))
        self.big_network[section_choice].mut_add_connection()

    def mut_random_filter(self):
        section_choice = random.choice(list(range(0, len(self.big_network), 2)))
        self.big_network[section_choice].mut_random_filter()

    def get_total_panes(self):
        panes_out = 0
        for layer in self.big_network:
            if type(layer) == Section and layer.pane_count_out > panes_out:
                panes_out = layer.pane_count_out
        return panes_out

    def __repr__(self):
        string = "{"
        string += "width: " + repr(self.width) + ", "
        string += "height: " + repr(self.height) + ", "
        string += "depth: " + repr(self.depth) + ", "
        string += "section_count: " + repr(self.section_count) + ", "
        string += "big_network: " + repr(self.big_network) + ", "
        string += "final_inputs_per_filter: " + repr(self.final_inputs_per_filter) + ", "
        string += "final_layer: " + repr(self.final_layer) + "}"
        return string


def mutate_individual(conv):
    x = random.random()
    if x < 0.05:
        conv.mut_add_filter()
    elif x < 0.1:
        conv.mut_add_connection()
    elif x < 0.4:
        conv.mut_random_filter()
    elif x < 0.95:
        conv.final_layer = ArtificialNetwork.mutate_individual(conv.final_layer)
    else:
        conv.mut_add_glcm()
    return conv


if __name__ == "__main__":
    print("new ind")
    best = (conv_network(224, 224, 3, 4, 4), )
    while True:
        print("new gen")
        convs = [best[0]]
        for k in range(10):
            convs.append(mutate_individual(dc(convs[-1])))
        inputs_bigger = np.zeros((2,3,224,224), dtype=float)

        species = "CANARY"
        for i in range(1,3):
            id = "00" + str(i) + ".jpg"
            inputs = np.array(Image.open("birds/train/" + species + "/" + id))
            working_outputs = []
            for colour in range(len(inputs[0][0])):
                working_outputs.append([])
                for y in range(len(inputs)):
                    working_outputs[-1].append([])
                    for x in range(len(inputs[0])):
                        working_outputs[-1][-1].append(inputs[y][x][colour] / 128 - 1)
            inputs_bigger[i-1] = np.array(working_outputs)
        #f = Filter(ArtificialNetwork.Network(9, 1, False), 3)
        #f.run(inputs_bigger)
        best = (convs[-1], convs[-1].run(inputs_bigger, show_as_images=True)[0][1])
        input()
        for c in convs:
            out = c.run(inputs_bigger)

            if out[0][1] > best[1]:
                best = (c, out[0][1])
            print(str(out))
        print(repr(convs[0]))
        input()

if __name__ == "frank":
    s = Section(len(working_outputs[0][0]), len(working_outputs[0]), len(working_outputs))
    mps = Max_Pool_Section()
    for i in range(10):
        s.mut_add_filter()
    out = s.run(working_outputs)
    toPlt = []
    for y in range(len(out[0])):
        toPlt.append([])
        for x in range(len(out[0][0])):
            toPlt[-1].append([out[len(out)-3][y][x], out[len(out)-2][y][x], out[len(out)-1][y][x]])
    plt.imshow(toPlt, interpolation="nearest")
    plt.show()

    out = mps.run(out)
    si = Section(len(out[0][0]), len(out[0]), len(out))
    for i in range(10):
        si.mut_add_filter()
    out = si.run(out)
    toPlt = []
    for y in range(len(out[0])):
        toPlt.append([])
        for x in range(len(out[0][0])):
            toPlt[-1].append([out[len(out)-3][y][x], out[len(out)-2][y][x], out[len(out)-1][y][x]])
    plt.imshow(toPlt, interpolation="nearest")
    plt.show()

    out = mps.run(out)
    si = Section(len(out[0][0]), len(out[0]), len(out))
    for i in range(10):
        si.mut_add_filter()
    out = si.run(out)
    toPlt = []
    for y in range(len(out[0])):
        toPlt.append([])
        for x in range(len(out[0][0])):
            toPlt[-1].append([out[len(out)-3][y][x], out[len(out)-2][y][x], out[len(out)-1][y][x]])
    plt.imshow(toPlt, interpolation="nearest")
    plt.show()

    out = mps.run(out)
    si = Section(len(out[0][0]), len(out[0]), len(out))
    for i in range(10):
        si.mut_add_filter()
    out = si.run(out)
    toPlt = []
    for y in range(len(out[0])):
        toPlt.append([])
        for x in range(len(out[0][0])):
            toPlt[-1].append([out[len(out)-3][y][x], out[len(out)-2][y][x], out[len(out)-1][y][x]])
    plt.imshow(toPlt, interpolation="nearest")
    plt.show()

