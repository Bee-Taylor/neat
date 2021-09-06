from ArtificialNetwork import Network
import ArtificialNetwork
import random
import time
from copy import deepcopy as dc
from multiprocessing import Pool
import sys


class Average_Pool:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def run(self, inputs):
        return [sum(inputs) / (self.width * self.height)]

    def __repr__(self):
        return "Average_Pool()"


class Max_Pool:
    def run(self, inputs):
        return [max(inputs)]

    def __repr__(self):
        return "Max_Pool()"


# This currently only works with 3x3 frames with stride of 3
def get_square(x, y):
    tmp = [[[i, j] for i in range(x - 1, x + 2)] for j in range(y - 1, y + 2)]
    output = []
    for i in tmp:
        output += i
    return output


class Convolutional_Network:
    def __init__(self, width, height, depth, layer_count, out_count, stride=1, diffusion_gradients=0,
                 new_individual=True, network=None, final_layer=None, filter_mapper=None,
                 max_pool_start=False, second_final_layer=False):
        self.width = width
        self.height = height
        self.depth = depth
        self.diffusion_gradients = diffusion_gradients
        self.sflBoolean = second_final_layer
        self.stride = stride
        initialFilterConnections = [[[1, i] for i in range(depth)]] + [[[1, 0]] for j in range(layer_count - 1)]
        if new_individual and self.stride > 1:
            if max_pool_start:
                self.network = [[[Max_Pool(), initialFilterConnections[i]]] for i in range(layer_count)]
            else:
                self.network = [[[Network(9 * len(initialFilterConnections[i]) + 1, 1, False,
                                          diffusion_gradients=self.diffusion_gradients),
                                  initialFilterConnections[i]]] for i in range(layer_count)]
        elif self.stride == 1:
            # TODO: Fix this
            self.network = []
            for i in range(layer_count * 2):
                if i % 2 == 0:
                    self.network.append(
                        [[Network(10, 1, False, diffusion_gradients=self.diffusion_gradients), [[1, 0]]]])
                else:
                    self.network.append([[Max_Pool(), None]])
        else:
            self.network = network
        self.out_count = out_count
        tmp_width = self.width
        tmp_height = self.height
        self.layer_count = layer_count
        for layer in range(len(self.network)):
            if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                tmp_width = tmp_width // 3 + (tmp_width % 3 != 0)
                tmp_height = tmp_height // 3 + (tmp_height % 3 != 0)
            elif self.stride == 2:
                tmp_width = tmp_width // 2
                tmp_height = tmp_height // 2
            elif self.stride == 1:
                tmp_width -= 2
                tmp_height -= 2
        if new_individual:
            self.final_layer = Network(tmp_width * tmp_height + 1, self.out_count, False,
                                       diffusion_gradients=self.diffusion_gradients)
            if second_final_layer:
                self.second_final_layer = Network(tmp_width * tmp_height + 1,
                                                  self.diffusion_gradients // self.out_count,
                                                  False, diffusion_gradients=0)
            else:
                self.second_final_layer = None
        else:
            self.final_layer = final_layer
        self.filter_mapper = filter_mapper
        if self.filter_mapper is None:
            self.filter_mapper = []
            tmp_height = self.height
            tmp_width = self.width
            for layer in range(len(self.network)):
                self.filter_mapper.append([])
                if type(self.network[layer][0][0]) != Max_Pool or self.stride != 1:
                    self.filter_mapper[-1] = [[[] for x in range(tmp_width + 6)] for y in range(tmp_height + 6)]
                    id = 0
                    count_to = tmp_height - 1
                    for y in range(1, count_to, self.stride):
                        for x in range(1, count_to, self.stride):
                            tmp = get_square(x, y)
                            counter = 0
                            for tmp_i in tmp:
                                self.filter_mapper[-1][tmp_i[1]][tmp_i[0]].append([id, counter])
                                counter += 1
                            id += 1
                if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                    tmp_width = tmp_width // 3 + (tmp_width % 3 != 0)
                    tmp_height = tmp_height // 3 + (tmp_height % 3 != 0)
                elif self.stride == 2:
                    tmp_width = tmp_width // 2
                    tmp_height = tmp_height // 2
                elif self.stride == 1:
                    tmp_width -= 2
                    tmp_height -= 2
        self.final_layer_copy = None
        self.mutation_log = ""

    def run(self, inputs):
        if len(inputs) != self.height:
            raise ValueError("We expected an input of height " + str(self.height) + " and this input is of height "
                             + str(len(inputs)))
        for row in inputs:
            if len(row) != self.width:
                raise ValueError("At least one of the rows is of width " + str(len(row)) +
                                 "when we expected all to be of width " + self.width)
        tmp_width = self.width
        tmp_height = self.height
        working_outputs = []
        for colour in range(self.depth):
            working_outputs.append([])
            for y in range(self.height):
                working_outputs[-1].append([])
                for x in range(self.width):
                    working_outputs[-1][-1].append(inputs[y][x][colour] / 128 - 1)
        counter = 0
        for layer in range(len(self.network)):
            working_outputs_tmp = []
            for filter in self.network[layer]:
                if type(filter[0]) is Max_Pool and self.stride == 1:
                    tmp_height_i = (tmp_height // 3 + (tmp_height % 3 != 0))
                    tmp_width_i = tmp_width // 3 + (tmp_width % 3 != 0)
                    input_buffer = [[None for i in range(tmp_height_i * tmp_width_i)]
                                    for j in range(len(self.network[layer - 1]))]
                    input_buffer_id = 0
                    y_ticker = 0
                    for y in range(tmp_height):
                        x_ticker = 0
                        ibi_start = input_buffer_id
                        for x in range(tmp_width):
                            if x_ticker == 3:
                                input_buffer_id += 1
                                x_ticker = 0
                            for input_filter in range(len(self.network[layer - 1])):
                                if input_buffer[input_filter][input_buffer_id] is None \
                                        or input_buffer[input_filter][input_buffer_id] < \
                                        working_outputs[input_filter][y][x]:
                                    input_buffer[input_filter][input_buffer_id] = working_outputs[input_filter][y][x]
                            x_ticker += 1
                        y_ticker += 1
                        if y_ticker == 3:
                            input_buffer_id += 1
                            y_ticker = 0
                        else:
                            input_buffer_id = ibi_start
                else:
                    working_outputs_tmp.append([])
                    filter_length_tmp = len(filter[1])
                    if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                        input_buffer = [[[0 for i in range(len(filter[1]))] for i in range(9)] for j in range(
                            (tmp_height // 3 + (tmp_height % 3 != 0)) * (tmp_width // 3 + (tmp_width % 3 != 0)))]
                    elif self.stride == 2:
                        input_buffer = [[[0 for i in range(len(filter[1]))] for i in range(9)] for j in
                                        range((tmp_height // 2) * (tmp_width // 2))]
                    elif self.stride == 1:
                        input_buffer = [[[0 for i in range(len(filter[1]))] for i in range(9)] for j in
                                        range((tmp_height - 2) * (tmp_width - 2))]
                    inner_start = time.time()
                    # print((tmp_height // 3 + (tmp_height%3 != 0)) * (tmp_width // 3 + (tmp_width%3 != 0)))
                    y_ticker = 0
                    input_buffer_id = 0
                    for y in range(len(working_outputs[0])):
                        for x in range(len(working_outputs[0][0])):
                            for target_filter in self.filter_mapper[layer][y][x]:
                                counter = 0
                                for input_filter in filter[1]:
                                    try:
                                        input_buffer[target_filter[0]][target_filter[1]][counter] += input_filter[0] * \
                                                                                                     working_outputs[
                                                                                                         input_filter[
                                                                                                             1]][y][x]
                                        counter += 1
                                    except IndexError:
                                        print("working outputs length = " + str(len(working_outputs)))
                                        print("input filter = " + str(input_filter))
                                        print("layer size 2 layers back = " + str(len(self.network[layer - 2])))
                                        print("input filters = " + str(filter[1]))
                                        print("original filter[1] length: " + str(filter_length_tmp))
                                        print("filter length now: " + str(len(filter[1])))
                                        print("input_buffer[0][0] = " + str(input_buffer[0][0]))
                                        print(self.mutation_log)
                                        raise IndexError
                    inner_start = time.time()
                    input_buffer_i = []
                    for ib in range(len(input_buffer)):
                        input_buffer_i.append([])
                        for depth in range(len(filter[1])):
                            for i in range(9):
                                input_buffer_i[ib].append(input_buffer[ib][i][depth])
                    try:
                        output_buffer = list(map(lambda x: x[0], map(filter[0].run, input_buffer_i)))
                    except ValueError as err:
                        print(self.mutation_log)
                        print(filter[1])
                        raise err
                # print(time.time() - inner_start)
                if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                    x_range = tmp_width // 3 + (tmp_width % 3 != 0)
                    y_range = tmp_height // 3 + (tmp_height % 3 != 0)
                elif self.stride == 2:
                    x_range = tmp_width // 2
                    y_range = tmp_height // 2
                elif self.stride == 1:
                    x_range = tmp_width - 2
                    y_range = tmp_height - 2
                if type(self.network[layer][0][0]) is Max_Pool and self.stride == 1:
                    for buffer in input_buffer:
                        working_outputs_tmp.append([])
                        for y in range(y_range):
                            working_outputs_tmp[-1].append([])
                            for x in range(x_range):
                                working_outputs_tmp[-1][-1].append(output_buffer[y * y_range + x])
                else:
                    for y in range(y_range):
                        working_outputs_tmp[-1].append([])
                        for x in range(x_range):
                            working_outputs_tmp[-1][-1].append(output_buffer[y * y_range + x])
            working_outputs = working_outputs_tmp
            if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                tmp_width = tmp_width // 3 + (tmp_width % 3 != 0)
                tmp_height = tmp_height // 3 + (tmp_height % 3 != 0)
            elif self.stride == 2:
                tmp_width = tmp_width // 2
                tmp_height = tmp_height // 2
            elif self.stride == 1:
                tmp_width -= 2
                tmp_height -= 2
        inputs = []
        for working_output in working_outputs:
            for y in working_output:
                for x in y:
                    inputs.append(x)
        # print("counter: " + str(counter), end=", total time: ")
        # print(time.time() - start_time)
        # print("\n\n\n")
        try:
            outputs = self.final_layer.run(inputs)
        except KeyError as err:
            print(self.final_layer.mutation_log)
            final_layer_log = open("final_layer_log.log", "w")
            final_layer_log.write(str(self.final_layer))
            final_layer_log.close()
            raise err
        return self.normalise_outputs(outputs)

    def normalise_outputs(self, outputs):
        outputs = [0 if x < 0 else x for x in outputs]
        if sum(outputs) > 0:
            outputs = [x / sum(outputs) for x in outputs]
        return outputs

    def show_example(self, inputs, correct_classification):
        if len(inputs) != self.height:
            raise ValueError("We expected an input of height " + str(self.height) + " and this input is of height "
                             + str(len(inputs)))
        for row in inputs:
            if len(row) != self.width:
                raise ValueError("At least one of the rows is of width " + len(row) +
                                 "when we expected all to be of width " + self.width)
        start_time = time.time()
        tmp_width = self.width
        tmp_height = self.height
        working_outputs = [inputs]
        counter = 0
        for layer in range(len(self.network)):
            working_outputs_tmp = []
            for filter in self.network[layer]:
                if type(filter[0]) is Max_Pool and self.stride == 1:

                    tmp_height_i = (tmp_height // 3 + (tmp_height % 3 != 0))
                    tmp_width_i = tmp_width // 3 + (tmp_width % 3 != 0)
                    input_buffer = [[None for i in range(tmp_height_i * tmp_width_i)]
                                    for j in range(len(self.network[layer - 1]))]
                    for y in range(tmp_height):
                        x_ticker = 0
                        ibi_start = input_buffer_id
                        for x in range(tmp_width):
                            if x_ticker == 3:
                                input_buffer_id += 1
                                x_ticker = 0
                            for input_filter in range(len(self.network[layer - 1])):
                                if input_buffer[input_filter][input_buffer_id] is None or input_buffer[input_filter][
                                    input_buffer_id] < working_outputs[input_filter][y][x]:
                                    input_buffer[input_filter][input_buffer_id] = working_outputs[input_filter][y][x]
                            x_ticker += 1
                        y_ticker += 1
                        if y_ticker == 3:
                            input_buffer_id += 1
                            y_ticker = 0
                        else:
                            input_buffer_id = ibi_start
                else:
                    working_outputs_tmp.append([])
                    filter_length_tmp = len(filter[1])
                    if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                        input_buffer = [[[0 for i in range(len(filter[1]))] for i in range(9)] for j in range(
                            (tmp_height // 3 + (tmp_height % 3 != 0)) * (tmp_width // 3 + (tmp_width % 3 != 0)))]
                    elif self.stride == 2:
                        input_buffer = [[[0 for i in range(len(filter[1]))] for i in range(9)] for j in
                                        range((tmp_height // 2) * (tmp_width // 2))]
                    elif self.stride == 1:
                        input_buffer = [[[0 for i in range(len(filter[1]))] for i in range(9)] for j in
                                        range((tmp_height - 2) * (tmp_width - 2))]
                    inner_start = time.time()
                    # print((tmp_height // 3 + (tmp_height%3 != 0)) * (tmp_width // 3 + (tmp_width%3 != 0)))
                    y_ticker = 0
                    input_buffer_id = 0
                    for y in range(len(working_outputs[0])):
                        for x in range(len(working_outputs[0][0])):
                            for target_filter in self.filter_mapper[layer][y][x]:
                                counter = 0
                                for input_filter in filter[1]:
                                    try:
                                        input_buffer[target_filter[0]][target_filter[1]][counter] \
                                            += input_filter[0] * working_outputs[input_filter[1]][y][x]
                                        counter += 1
                                    except IndexError:
                                        print("working outputs length = " + str(len(working_outputs)))
                                        print("input filter = " + str(input_filter))
                                        print("layer size 2 layers back = " + str(len(self.network[layer - 2])))
                                        print("input filters = " + str(filter[1]))
                                        print("original filter[1] length: " + str(filter_length_tmp))
                                        print("filter length now: " + str(len(filter[1])))
                                        print("input_buffer[0][0] = " + str(input_buffer[0][0]))
                                        print(self.mutation_log)
                                        raise IndexError
                    input_buffer_i = []
                    for ib in range(len(input_buffer)):
                        input_buffer_i.append([])
                        for depth in range(len(filter[1])):
                            for i in range(9):
                                input_buffer_i[ib].append(input_buffer[ib][i][depth])
                    try:
                        output_buffer = list(map(lambda x: x[0], map(filter[0].run, input_buffer_i)))
                    except ValueError as err:
                        print(self.mutation_log)
                        print(filter[1])
                        raise err
                # print(time.time() - inner_start)
                if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                    x_range = tmp_width // 3 + (tmp_width % 3 != 0)
                    y_range = tmp_height // 3 + (tmp_height % 3 != 0)
                elif self.stride == 2:
                    x_range = tmp_width // 2
                    y_range = tmp_height // 2
                elif self.stride == 1:
                    x_range = tmp_width - 2
                    y_range = tmp_height - 2
                if type(self.network[layer][0][0]) is Max_Pool and self.stride == 1:
                    for buffer in input_buffer:
                        working_outputs_tmp.append([])
                        for y in range(y_range):
                            working_outputs_tmp[-1].append([])
                            for x in range(x_range):
                                working_outputs_tmp[-1][-1].append(output_buffer[y * y_range + x])
                else:
                    for y in range(y_range):
                        working_outputs_tmp[-1].append([])
                        for x in range(x_range):
                            working_outputs_tmp[-1][-1].append(output_buffer[y * y_range + x])
            working_outputs = working_outputs_tmp
            if self.stride == 3 or (self.stride == 1 and type(self.network[layer][0][0]) == Max_Pool):
                tmp_width = tmp_width // 3 + (tmp_width % 3 != 0)
                tmp_height = tmp_height // 3 + (tmp_height % 3 != 0)
            elif self.stride == 2:
                tmp_width = tmp_width // 2
                tmp_height = tmp_height // 2
            elif self.stride == 1:
                tmp_width -= 2
                tmp_height -= 2
        inputs = []
        for working_output in working_outputs:
            for y in working_output:
                for x in y:
                    inputs.append(x)
        self.final_layer.learn([0 for i in range(
            (self.diffusion_gradients * correct_classification) // self.out_count)] + self.second_final_layer.run(
            inputs) + [0 for i in range((self.diffusion_gradients * (
                                           self.out_count - correct_classification - 1)) // self.out_count)])

    def mut_add_filter(self):
        layer = random.randint(0, len(self.network) - 1)
        if type(self.network[layer][0][0]) is Max_Pool and self.stride == 1:
            return
        if layer != 0:
            if type(self.network[layer - 1][0][0]) is Max_Pool and self.stride == 1:
                val = -2
            else:
                val = -1
            try:
                self.network[layer].append(
                    [Network(10, 1, False, diffusion_gradients=self.diffusion_gradients),
                     [[random.gauss(0, 0.333), random.randint(0, len(self.network[layer + val]) - 1)]]])
            except IndexError:
                print(self.network)
                print("network length = " + str(len(self.network)))
                print("layer = " + str(layer))
                print("val = " + str(val))
        else:
            self.network[layer].append(
                [Network(10, 1, False, diffusion_gradients=self.diffusion_gradients),
                 [[random.gauss(0, 0.333), random.randint(0, self.depth - 1)]]])
        flag = True
        if layer != len(self.network) - 1:
            if type(self.network[layer + 1][0][0]) is Max_Pool and self.stride == 1:
                flag = not (layer >= len(self.network) - 2)
                val = 2
            else:
                val = 1
            if flag:
                filter = random.choice(self.network[layer + val])
                filter[1].append([random.gauss(0, 0.1), len(self.network[layer]) - 1])
                filter[0].add_inputs(9)
        if layer == len(self.network) - 1 or not flag:
            self.final_layer.add_inputs(self.final_layer.original_in_count - 1)
            if self.second_final_layer is not None:
                self.second_final_layer.add_inputs(self.final_layer.original_in_count - 1)
        self.mutation_log += "add filter, "

    def mut_add_connection(self):
        layer = random.randint(1, len(self.network) - 1)
        if type(self.network[layer][0][0]) is Max_Pool and self.stride == 1:
            return
        choice = random.randint(0, len(self.network[layer]) - 1)
        if type(self.network[layer - 1][0][0]) is Max_Pool and self.stride == 1:
            val = -2
        else:
            val = -1
        if len(self.network[layer][choice][1]) < len(self.network[layer + val]):
            tmp = [i for i in range(len(self.network[layer + val]))]
            for j in self.network[layer][choice][1]:
                if j[1] in tmp:
                    tmp.remove(j[1])
            self.network[layer][choice][1].append([random.gauss(0, 0.333), random.choice(tmp)])
            self.network[layer][choice][0].add_inputs(9)
        self.mutation_log += "add connection, "

    def mut_some_filters(self, prob):
        for layer in self.network:
            for filter in layer:
                if random.random() < prob and type(filter[0]) is not Max_Pool and type(filter[0]) is not Average_Pool:
                    ArtificialNetwork.mutate_individual(filter[0])
        self.mutation_log += "mutate filters, "

    def learn(self, reward_signals):
        self.final_layer.learn(reward_signals)

    def revert(self):
        self.final_layer.revert()

    def mut_connection_weights(self, prob):
        for layer in self.network:
            for filter in layer:
                for input_filter in filter[1]:
                    if prob > random.random():
                        input_filter[0] += random.gauss(0, 0.05)
        self.mutation_log += "mutate weights, "

    def __repr__(self):
        to_return = "Convolutional_Network("
        to_return += repr(self.width)
        to_return += ", "
        to_return += repr(self.height)
        to_return += ", "
        to_return += repr(self.depth)
        to_return += ", "
        to_return += repr(self.layer_count)
        to_return += ", "
        to_return += repr(self.out_count)
        to_return += ", diffusion_gradients="
        to_return += repr(self.diffusion_gradients)
        to_return += ", new_individual=False, network="
        to_return += repr(self.network)
        to_return += ", final_layer="
        to_return += repr(self.final_layer)
        to_return += ")"
        return to_return


def copy(conv_network):
    if type(conv_network) == list:
        return list(map(copy, conv_network))
    filter_mapper = conv_network.filter_mapper
    final_layer = dc(conv_network.final_layer)
    network = dc(conv_network.network)
    width, height, depth, layer_count, out_count, stride, diffusion_gradients = \
        conv_network.width, conv_network.height, conv_network.depth, conv_network.layer_count, \
        conv_network.out_count, conv_network.stride, conv_network.diffusion_gradients
    to_return = Convolutional_Network(width, height, depth, layer_count, out_count, stride=stride,
                                      diffusion_gradients=diffusion_gradients, new_individual=True, network=network,
                                      final_layer=final_layer, filter_mapper=filter_mapper)
    to_return.mutation_log = conv_network.mutation_log
    return to_return


def mutate_individual(individual):
    if type(individual) == list or type(individual) == tuple:
        for ind_i in individual:
            mutate_individual(ind_i)
    choice = random.randint(0, 5)
    if choice == 0:
        individual.mut_add_filter()
    elif choice == 1:
        individual.mut_add_connection()
    elif choice == 2:
        individual.mut_some_filters(0.05)
    elif choice == 3:
        ArtificialNetwork.mutate_individual(individual.final_layer)
    elif choice == 4 and individual.sflBoolean:
        ArtificialNetwork.mutate_individual(individual.second_final_layer)
    else:
        individual.mut_connection_weights(0.05)
    return individual
