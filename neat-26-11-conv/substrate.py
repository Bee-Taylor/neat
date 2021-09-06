#import snakeProblem
import numpy
import CPPN
import random
import time
from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
from copy import deepcopy as dc
from workers import worker
import math
#import testing
import t_maze
#import foraging_task
# input()

# random.seed(2)


class Testing_Cppn:
    def __init__(self, output_value):
        self.output_value = output_value

    def run(self, inputs):
        return [self.output_value]


class Network:
    def __init__(self, nodes, in_count, out_count, tmp, recurrent_connections, fake_input_count=0,
                 fake_output_count=0, plasticity = False, dsp_rule = None, diffusion_gradients = 0):
        self.nodes = nodes
        self.recurrent_connections = []
        self.correct_node_order = None
        self.in_count = in_count
        self.out_count = out_count
        self.fake_input_count = fake_input_count
        self.fake_output_count = fake_output_count
        self.tmp = tmp
        self.recurrent_connections = [0 for i in range(recurrent_connections)]
        self.plasticity = plasticity
        self.dsp_rule = dsp_rule
        self.get_runnable()
        self.nat_data = [[] for node in nodes]
        self.diffusion_gradients = diffusion_gradients
        for node in self.nodes:
            for connection in node[0]:
                for reward in range(diffusion_gradients):
                    if connection[4][reward]**2 < 1:
                        connection[4][reward] = 0

    def get_runnable_old(self):
        for node in self.nodes:
            node[4] = 0
        self.correct_node_order = []
        not_added = []
        finished = False
        while not finished:
            node_added = False
            for node in range(len(self.nodes)):
                if self.nodes[node][4] >= self.nodes[node][3][
                        0] and node not in self.correct_node_order and node not in not_added:
                    if self.nodes[node][3][0] == 0 and node >= self.in_count:
                        not_added.append(node)
                    else:
                        node_added = True
                        self.correct_node_order.append(node)
                        for connection in self.nodes[node][0]:
                            try:
                                self.nodes[connection[2]][4] += 1
                            except IndexError:
                                print("whoops")
            if not node_added:
                raise ValueError("This ANN includes an infinite loop")
            if len(self.correct_node_order) + len(not_added) == len(self.nodes):
                finished = True

    def get_runnable(self):
       # print("entering get runnable")
        self.correct_node_order = [i for i in range(self.in_count)]
        non_inputs = self.nodes[self.in_count:]
        current_min_x = 1
        last_min_x = -1
        counter = 0
        while len(self.correct_node_order) < len(self.nodes):
            #print(counter)
            previous_length = len(self.correct_node_order)
            for node in non_inputs:
                if last_min_x < node[-1][0] < current_min_x:
                    current_min_x = node[-1][0]
            for node in range(len(self.nodes)):
                if self.nodes[node][-1][0] == current_min_x and node not in self.correct_node_order:
                    self.correct_node_order.append(node)
            if len(self.correct_node_order) == previous_length:
                raise RuntimeError("Somethings gone wrong")
            last_min_x = current_min_x
            current_min_x = 1
        for node in self.nodes:
            for connection in node[0]:
                if connection[3] is not None:
                    self.nodes[connection[2]][3].append(connection[3])
       # print("exiting get runnable")


    def run(self, inputs):
        #print("running")
        for node in self.nodes:
            node[2] = 0
        if len(inputs) != self.in_count - 1:
            raise ValueError("This is not the correct amount of inputs, i expected " + str(
                self.in_count - 1) + " inputs and I received " + str(len(inputs)))
        for i in range(self.in_count - 1):
            self.nodes[i][2] = inputs[i]
        #for j in range(self.in_count - 1, self.in_count + self.fake_input_count):
        #    self.nodes[j][2] = 0
        for node in self.correct_node_order:
            #print(node)
            ##if node == 41:
            #    print("here")
            for recurrent in self.nodes[node][3]:
                self.nodes[node][2] += self.recurrent_connections[recurrent]
            if self.in_count <= node <= len(self.nodes) - self.out_count:
                self.nodes[node][2] = CPPN.ACTIVATION_FUNCTIONS["tanh"](self.nodes[node][2])
            if self.plasticity:
                self.nat_data[node].append(0 if self.nodes[node][2]<0 else 1)
            #self.nodes[node][2] = self.nodes[node][1][0] * CPPN.ACTIVATION_FUNCTIONS["gauss"](self.nodes[node][2]) + \
                                  #self.nodes[node][1][1] * CPPN.ACTIVATION_FUNCTIONS["relu"](self.nodes[node][2]) + \
                                  #self.nodes[node][1][2] * CPPN.ACTIVATION_FUNCTIONS["tanh"](self.nodes[node][2])
            #if self.nodes[node][2] > 100:
            #    self.nodes[node][2] = 100
            #elif self.nodes[node][2] < -100:
            #    self.nodes[node][2] = -100
            for connection in self.nodes[node][0]:
                if connection[3] is not None:
                    self.recurrent_connections[connection[3]] = self.nodes[node][2] * connection[1]
                else:
                    self.nodes[connection[2]][2] += self.nodes[node][2] * connection[1]
        outputs = []
        for output in self.nodes[-self.out_count - self.fake_output_count:]:
            outputs.append(output[2])
        #print("finished running")
        if self.fake_output_count > 0:
            return outputs[:-self.fake_output_count]
        else:
            return outputs

    def learn(self, reward_signals):
        if len(reward_signals) != self.diffusion_gradients:
            raise ValueError(str(reward_signals) + " were passed in to the network, we expected " +
                             str(self.diffusion_gradients) + " signals.")
        for node in range(len(self.nodes)):
            for connection in self.nodes[node][0]:
                nat = [0 for i in range(4)]
                for line in range(len(self.nat_data[0])):
                    nat[self.nat_data[node][line]*2 + self.nat_data[connection[2]][line]] += 1
                for reward in range(len(reward_signals)):
                    if reward_signals[reward] != 0:
                        connection[1] += 0.1 * connection[4][reward] * self.dsp_rule.get_delta_w(nat, reward_signals[reward])
        self.nat_data = [[] for node in self.nodes]




def my_linspace(bottom, top, iter_count):
    if iter_count == 0:
        return []
    to_return = [bottom]
    iterator = float(top - bottom) / float(iter_count)
    for i in range(iter_count - 1):
        to_return.append(to_return[-1] + iterator)
    return to_return


# assumes the target should be square with nodes spread as evenly as possible
# only works for square numbers
def my_2d_linspace(iter_count):
    iter_count_added = 0
    sqrt = math.sqrt(iter_count)
    if sqrt < round(sqrt):
        number_in_x = round(sqrt)
        number_in_y = round(sqrt) - 1
    elif sqrt > round(sqrt):
        number_in_x = round(sqrt) + 1
        number_in_y = round(sqrt)
    else:
        number_in_x = number_in_y = round(sqrt)
    to_return = []
    for x in range(0, number_in_x - 1):
        for y in range(0, number_in_y):
            # if number_per_row % 2 == 0:
            #    to_return.append([(x/(number_per_row/2)) - 1 + (1/number_per_row), (y/(number_per_row/2)) - 1 + (1/number_per_row)])
            # else:
            to_return.append([(x / (number_in_x / 2)) - 1, (y / (number_in_y / 2)) - 1])
    x = number_in_x - 1
    for y in range(0, iter_count - len(to_return)):
        to_return.append([(x / (number_in_x / 2)) - 1, (y / (number_in_y / 2)) - 1])
    return to_return, iter_count_added


def split_in_4(x0, y0, x1, y1):
    x_av = (x0+x1)/2
    y_av = (y0+y1)/2
    return [[x0, y0, x_av, y_av], [x_av, y0, x1, y_av], [x0, y_av, x_av, y1], [x_av, y_av, x1, y1]]

class Substrate:
    def __init__(self, x, y, in_count, out_count, activation, fitness_functions, population_size, number_of_species,
                 number_of_generations, diffusion_gradients = 0, plasticity = False):
        self.x = x
        self.y = y
        self.in_count = in_count
        self.out_count = out_count
        self.activation = activation
        self.fitness_functions = fitness_functions
        self.networks = []
        self.fitnesses = []
        self.max_gens = number_of_generations
        self.population_size = population_size
        #self.cppn_database = CPPN.CPPN_Database(population_size, in_count - 1, out_count,
        #                                        number_of_species=number_of_species, plasticity = plasticity,
        #                                        diffusion_gradients = diffusion_gradients)
        self.choice = 0
        self.gen = 0
        if plasticity and diffusion_gradients == 0:
            raise ValueError("This system does not support plasticity without diffusion gradients")
        self.diffusion_gradients = diffusion_gradients
        self.plasticity = plasticity

    def explore_node(self, next_row_base_node, coordinates, additional_inputs, in_count_for_future_nodes, nodes, cppn):
        next_node_id = next_row_base_node
        for next_node in my_2d_linspace(self.y)[0]:
            if next_node == [0, 0]:
                continue
            weight = cppn.run([*input, *next_node, *additional_inputs])[0]
            if weight > 0.2 or weight < -0.2:
                if weight > 0:
                    weight = min(3, 0.5 * (weight - 0.2))
                else:
                    weight = max(-3, 0.5 * (weight + 0.2))
                nodes[-1][0].append([0, weight, next_node_id])
                if next_node_id not in in_count_for_future_nodes:
                    in_count_for_future_nodes[next_node_id] = 1
                else:
                    in_count_for_future_nodes[next_node_id] += 1
            next_node_id += 1

    def generate_activation(self, c0, c1, c2):
        def to_return(x): return c0 * (x ** 2) + c1 * x + c2

        return to_return


    #i need to figure out how it can ever connect to outputs. this won't currently work.
    def convert_cppn_into_network(self, cppn, *additional_inputs):
        #print("entering")
        nodes = []
        recurrent_connections = 0
        node_coordinates = [[-1, x] for x in my_linspace(-1, 1, self.in_count)]
        node_coordinate_index = 0
        in_count_for_future_nodes = {}
        init_depth = 5
        max_depth = 3
        bandpruning_threshold = 0.3
        variance_treshold = 0.03
        division_threshold = 0.03
        iteration_level = 1
        id_to_index = []
        start_time = time.time()
        weight_threshold = 0.5
        while node_coordinate_index < len(node_coordinates):
            node = node_coordinates[node_coordinate_index]
            nodes.append([[], [None], 0,  [], 0, node])
            # create quadtree
            quadtree = [-1, -1, 1, 1, [], [], [], [], 0, cppn.run(node + [0, 0, *additional_inputs])[0]]
            first = True
            stack = []
            #print("beginning first stack")
            while first or len(stack) > 0:
                first = False
                quadtree_tmp = quadtree
                if len(stack) > 0:
                    next = stack.pop()
                    while next >= 3 and len(stack) > 0:
                        next = stack.pop()
                    if next >= 3:
                        break
                    next += 1
                    stack.append(next)
                #if len(stack) >= max_depth:
                #    stack.pop()
                for level in stack:
                    if level > 3:
                        stack.pop()
                        if len(stack) != 0:
                            stack[-1] += 1
                            continue
                        else:
                            break
                    quadtree_tmp = quadtree_tmp[4+level]
                splits = split_in_4(*quadtree_tmp[:4])
                index = 0
                for split in splits:
                    quadtree_tmp[4 + index] = [*split, [], [], [], [], 0, cppn.run(node + [(split[0] + split[2]) / 2,
                                                                               (split[1] + split[3]) / 2, *additional_inputs])[0]]
                    index += 1
                quadtree_tmp[8] = numpy.var(list(map(lambda x: x[-1], quadtree_tmp[4:8])))
                if quadtree_tmp[8] > division_threshold and len(stack) < max_depth:
                    stack.append(-1)
            # prune the quadtree
            first = True
            #print("beginning second stack")
            while first or len(stack) > 0:
                first = False
                quadtree_tmp = quadtree
                if len(stack) > 0:
                    next = stack.pop()
                    while next >= 3 and len(stack) > 0:
                        next = stack.pop()
                    if next >= 3:
                        break
                    next += 1
                    stack.append(next)
                #if len(stack) >= max_depth:
                #    stack.pop()
                for level in stack:
                    if level < 0 or level > 3:
                        stack.pop()
                        if len(stack) != 0:
                            stack[-1] += 1
                            continue
                        else:
                            break
                    quadtree_tmp = quadtree_tmp[4+level]
                if len(quadtree_tmp) == 0:
                    continue
                if quadtree_tmp[8] < variance_treshold:
                    for i in range(4):
                        quadtree_tmp[4+i] = []
                if len(stack) < max_depth:
                    stack.append(-1)
            # perform band-pruning: not yet implemented
            #print("beginning third stack")
            target_node_stacks = []
            target_node_coordinates = []
            first = True
            while first or len(stack) > 0:
                first = False
                quadtree_tmp = quadtree
                if len(stack) > 0:
                    next = stack.pop()
                    while next >= 3 and len(stack) > 0:
                        next = stack.pop()
                    if next >= 3:
                        break
                    next += 1
                    stack.append(next)
                #if len(stack) >= max_depth:
                #    stack.pop()
                for level in stack:
                    if level < 0 or level > 3:
                        stack.pop()
                        if len(stack) != 0:
                            stack[-1] += 1
                            continue
                        else:
                            break
                    quadtree_tmp = quadtree_tmp[4+level]
                if len(quadtree_tmp) == 0:
                    continue
                if len(quadtree_tmp[4]) == 0:
                    target_node_stacks.append(dc(stack))
                    target_node_coordinates.append([(quadtree_tmp[0] + quadtree_tmp[2]) / 2, (quadtree_tmp[1] + quadtree_tmp[3]) / 2])
                if len(stack) < max_depth:
                    stack.append(-1)
                # note to self of what to do here: you need to find the leaf nodes, convert their stack value in to an id
                # written on window how
                # then add coordinates of added node to list of unexplored nodes (if not already in there)
            for target_node_coordinate in target_node_coordinates:
                recurrent_data = None
                if target_node_coordinate[0] <= nodes[-1][5][0]:   
                    recurrent_data = recurrent_connections
                    recurrent_connections += 1
                if target_node_coordinate not in node_coordinates:
                    nodes[-1][0].append([0, cppn.run(node + target_node_coordinate + [*additional_inputs])[0],
                                         len(node_coordinates), recurrent_data])
                    node_coordinates.append(target_node_coordinate)
                else:
                    nodes[-1][0].append([0, cppn.run(node + target_node_coordinate + [*additional_inputs])[0],
                                         node_coordinates.index(target_node_coordinate), recurrent_data])
                if self.plasticity:
                    nodes[-1][0][-1].append(cppn.run(node + target_node_coordinate + [*additional_inputs])[1:])
            node_coordinate_index += 1
        #print("output time")
        output_node_coordinates = [[1, x] for x in my_linspace(-1, 1, self.out_count)]
        output_nodes = []
        for output_coord in output_node_coordinates:
            output_nodes.append([[], [None], 0, [], 0, output_coord])
        for node in nodes:
            if node[5][0] >= 0:
                index = 0
                for output_coord in output_node_coordinates:
                    weight = cppn.run(node[5] + output_coord + [*additional_inputs])[0]
                    if abs(weight) > weight_threshold:
                        node[0].append([0, weight, len(nodes) + index, None, cppn.run(node[5] + output_coord + [*additional_inputs])[1:]])
                    index += 1
        nodes += output_nodes
        end_time = time.time() - start_time
        #print("exiting conversion")
        return Network(nodes, self.in_count, self.out_count, in_count_for_future_nodes, recurrent_connections,
                       plasticity=self.plasticity, dsp_rule=cppn.dsp_rule, diffusion_gradients=self.diffusion_gradients)






    def convert_cppn_into_network_old(self, cppn, *additional_inputs):
        nodes = []
        current_x = - 1 - (2 / self.x)
        recurrent_connecions = 0
        inputs, fake_inputs = my_2d_linspace(self.in_count)
        next_row_base_node = self.in_count + fake_inputs
        in_count_for_future_nodes = {}
        for input in inputs:
            nodes.append([[], [0, 0, 1], 0, [0, None], 0])
            next_node_id = next_row_base_node
            for next_node in my_2d_linspace(self.y)[0]:
                if next_node == [0, 0]:
                    continue
                weight = cppn.run([*input, *next_node, *additional_inputs])[0]
                if weight > 0.2 or weight < -0.2:
                    if weight > 0:
                        weight = min(3, 0.5 * (weight - 0.2))
                    else:
                        weight = max(-3, 0.5 * (weight + 0.2))
                    nodes[-1][0].append([0, weight, next_node_id])
                    if next_node_id not in in_count_for_future_nodes:
                        in_count_for_future_nodes[next_node_id] = 1
                    else:
                        in_count_for_future_nodes[next_node_id] += 1
                next_node_id += 1
        next_row_base_node += self.y + my_2d_linspace(self.y)[1]
        for current_x_index in range(self.x):
            nodes_coords, added_nodes = my_2d_linspace(self.y)
            for node in nodes_coords:
                if node == [0, 0]:
                    continue
                if len(nodes) not in in_count_for_future_nodes:
                    in_count_for_future_nodes[len(nodes)] = 0
                activation = cppn.run([0, 0, *node, *additional_inputs])[
                             2 * (self.x + 1) + 3 * current_x_index:2 * (self.x + 1) + 3 * current_x_index + 3]
                # print(activation(2))
                nodes.append([[], activation, 0, [in_count_for_future_nodes[len(nodes)], None], 0])
                if nodes[-1][3][0] > 0:
                    next_node_id = next_row_base_node
                    for next_node in my_2d_linspace(self.y)[0]:
                        if next_node == [0, 0]:
                            continue
                        weight = cppn.run([*node, *next_node, *additional_inputs])[current_x_index * 2]
                        if weight > 0.2 or weight < -0.2:
                            if weight > 0:
                                weight = min(3, 0.5 * (weight - 0.2))
                            else:
                                weight = max(-3, 0.5 * (weight + 0.2))
                            nodes[-1][0].append([0, weight, next_node_id])
                            if next_node_id not in in_count_for_future_nodes:
                                in_count_for_future_nodes[next_node_id] = 1
                            else:
                                in_count_for_future_nodes[next_node_id] += 1
                        next_node_id += 1
                if current_x_index > 0:
                    weight = cppn.run([*node, *node, *additional_inputs])[2 * (current_x_index - 1)]
                    if weight > 0.2 or weight < -0.2:
                        if weight > 0:
                            weight = min(3, 0.5 * (weight - 0.2))
                        else:
                            weight = max(-3, 0.5 * (weight + 0.2))
                        nodes[-1][0].append([0, weight, len(nodes) - 1])
                        nodes[-1][3][1] = recurrent_connecions
                        recurrent_connecions += 1
            node = [0, 0]
            next_node_id = next_row_base_node
            for next_node in my_2d_linspace(self.y)[0]:
                if next_node == [0, 0]:
                    continue
                weight = cppn.run([*node, *next_node, *additional_inputs])[(2 * current_x_index) + 1]
                if weight > 0.2 or weight < -0.2:
                    if weight > 0:
                        weight = min(3, 0.5 * (weight - 0.2))
                    else:
                        weight = max(-3, 0.5 * (weight + 0.2))
                    nodes[self.in_count - 1][0].append([0, weight, next_node_id])
                    if next_node_id not in in_count_for_future_nodes:
                        in_count_for_future_nodes[next_node_id] = 1
                    else:
                        in_count_for_future_nodes[next_node_id] += 1
                next_node_id += 1
            next_row_base_node += self.y + added_nodes
        current_x_index += 1
        for node in my_2d_linspace(self.y)[0]:
            if node == [0, 0]:
                continue
            if len(nodes) not in in_count_for_future_nodes:
                in_count_for_future_nodes[len(nodes)] = 0
            activation = cppn.run([0, 0, *node, *additional_inputs])[
                         2 * (self.x + 1) + 3 * current_x_index:2 * (self.x + 1) + 3 * current_x_index + 3]
            nodes.append([[], activation, 0, [in_count_for_future_nodes[len(nodes)], None], 0])
            if nodes[-1][3][0] > 0:
                next_node_id = next_row_base_node
                outputs, fake_outputs = my_2d_linspace(self.out_count)
                for next_node in outputs:
                    weight = cppn.run([*node, *next_node, *additional_inputs])[2 * current_x_index]
                    if weight > 0.2 or weight < -0.2:
                        if weight > 0:
                            weight = min(3, 0.5 * (weight - 0.2))
                        else:
                            weight = max(-3, 0.5 * (weight + 0.2))
                        nodes[-1][0].append([0, weight, next_node_id])
                        if next_node_id not in in_count_for_future_nodes:
                            in_count_for_future_nodes[next_node_id] = 1
                        else:
                            in_count_for_future_nodes[next_node_id] += 1
                    next_node_id += 1
                weight = cppn.run([*node, *node, *additional_inputs])[2 * (current_x_index - 1)]
                if weight > 0.2 or weight < -0.2:
                    if weight > 0:
                        weight = min(3, 0.5 * (weight - 0.2))
                    else:
                        weight = max(-3, 0.5 * (weight + 0.2))
                    nodes[-1][0].append([0, weight, len(nodes) - 1])
                    nodes[-1][3][1] = recurrent_connecions
                    recurrent_connecions += 1
        node = [0, 0]
        next_node_id = next_row_base_node
        outputs, fake_outputs = my_2d_linspace(self.out_count)
        for next_node in outputs:
            weight = cppn.run([*node, *next_node, *additional_inputs])[(2 * current_x_index) + 1]
            if weight > 0.2 or weight < -0.2:
                if weight > 0:
                    weight = min(3, 0.5 * (weight - 0.2))
                else:
                    weight = max(-3, 0.5 * (weight + 0.2))
                nodes[self.in_count - 1][0].append([0, weight, next_node_id])
                if next_node_id not in in_count_for_future_nodes:
                    in_count_for_future_nodes[next_node_id] = 1
                else:
                    in_count_for_future_nodes[next_node_id] += 1
            next_node_id += 1
        outputs, fake_outputs = my_2d_linspace(self.out_count)
        for node in range(self.out_count + fake_outputs):
            if len(nodes) not in in_count_for_future_nodes:
                nodes.append([[], [0, 0, 1], 0, [0, None], 0])
            else:
                nodes.append([[], [0, 0, 1], 0, [in_count_for_future_nodes[len(nodes)], None], 0])
        return Network(nodes, in_count, out_count, in_count_for_future_nodes, recurrent_connecions,
                       fake_input_count=fake_inputs, fake_output_count=fake_outputs)

    def new_population(self, cppn_list):
        with Pool(processes=16) as pool:
            self.networks = list(pool.map(self.convert_cppn_into_network, cppn_list))

    def evaluate_fitnesses(self):
        with Pool(processes=16) as pool:
            return list(pool.map(self.fitness_function, self.networks))

    def new_pop_and_eval(self):
        cppn_list = self.cppn_database.get_cppns()
        #cppn_list = dc(cppn_list)
        queue_out = Queue()
        queue_in = Queue()
        new_networks = Queue()
        process_count = 4
        self.networks = [cppn_list[i] for i in range(len(cppn_list))]
        processes = [Process(target=worker, args=(queue_out, queue_in, new_networks)) for i in range(process_count)]
        # decide on fitness function for this run
        # self.choice = random.randint(0, len(self.fitness_functions)-1)
        if len(self.fitness_functions) > 1:
            self.choices = list(range(len(self.fitness_functions)))[1:]
        else:
            self.choices = list(range(len(self.fitness_functions)))
        fitnesses = [None for i in range(len(cppn_list))]
        for process in processes:
            process.daemon = True
            process.start()
        random.shuffle(self.choices)
        if self.gen < 10:
            self.choices = [0] + self.choices[:1]
        else:
            self.choices = self.choices[:1]
        #self.choices = [1 for i in range(5)]
        index = 0
        for self.choice in self.choices:
            index += 1
            additional_input = [
                2 * (self.choice / len(self.fitness_functions)) - 1]  # [0 for i in range(len(self.fitness_functions))]
            # additional_input[self.choice] = 1
            fitness_function = self.fitness_functions[self.choice]
            for cppn in range(len(cppn_list)):
                queue_out.put(
                    [cppn, fitness_function, cppn_list[cppn]])
            #if index == len(self.choices):
            #    for i in range(len(cppn_list)):
            #        net = new_networks.get()
            #        # for i in range(1):
                    #    queue_out.put([net[0], fitness_function, net[1]])
            #        self.networks[net[0]] = net[1]
        for i in range(len(cppn_list) * len(self.choices)):
            output = queue_in.get()
            if fitnesses[output[0]] is None:
                fitnesses[output[0]] = output[1]
            else:
                fitnesses[output[0]] += output[1]
        for process in range(process_count):
            queue_out.put(None)

        for process in processes:
            process.terminate()
            process.join()
        self.fitnesses = fitnesses
        self.cppn_database.distribute_fitnesses(self.fitnesses)

    def __getitem__(self, index):
        return self.networks[index]

    def run(self):
#        foraging_task.new_gen()
        queue = Queue()
        Process(target=testing.walker, args=(queue,)).start()
        print("generation  species          std           max            avg       pop_size          min         "
              "time         choice")
        with open("output_file.csv", "a") as out:
            out.write("new evaluation")
        for gen in range(num_gens):
            self.gen = gen
            start_time = time.time()
            # substrate.new_population(cppn_database.get_cppns())
            # fitnesses = substrate.evaluate_fitnesses()
            self.new_pop_and_eval()
            self.cppn_database.distribute_meta(gen, round(time.time() - start_time), self.choice)
            print(str(self.cppn_database))
            with open("output_file.csv", "a") as out:
                out.write(repr(self.cppn_database))
            self.cppn_database.save_to_file()
            self.cppn_database.new_generation(self.fitnesses)
 #           foraging_task.new_gen()
            while not queue.empty():
                queue.get()
            # snakeProblem.displayStrategyRun(self.networks[self.fitnesses.index(max(self.fitnesses))], choice=self.choice)
            if gen < num_gens - 1:
                self.choice = 0
                queue.put((gen < num_gens - 1, dc(substrate[self.fitnesses.index(max(self.fitnesses))]), self.choice))

    def get_best_ind(self):
        return substrate[self.fitnesses.index(max(self.fitnesses))]


def f_function(network):
    outputs = None
    outputs_i = None
    for j in range(100):
        input_vals = [(j - j / 2) / 100 for i in range(network.in_count - 1)]
        output = network.run(input_vals)
        if outputs is None:
            outputs = [0 for i in range(len(output))]
            outputs_i = [0 for i in range(len(output))]
        #if output.index(max(output)) not in outputs:
        #    outputs.append(output.index(max(output)))
        outputs[output.index(max(output))] += 1
        outputs_i[output.index(min(output))] += 1

    #outputs.sort()
    return (min(outputs) + min(outputs_i)) * 5


def second_f_function(network):
    inputs = [random.choice([0, 1, 2, 3, 4]), random.choice([0, 1, 2, 3, 4]), random.choice([0, 1, 2, 3, 4]),
              random.choice([0, 1, 2, 3, 4])]
    outputs = network.run(inputs)
    if inputs[0] < inputs[1] and outputs.index(max(outputs)) == 0:
        return 1
    if inputs[1] < inputs[0] and outputs.index(max(outputs)) == 1:
        return 1
    if inputs[2] < inputs[3] and outputs.index(max(outputs)) == 2:
        return 1
    if inputs[3] < inputs[2] and outputs.index(max(outputs)) == 3:
        return 1
    return 0


if __name__ == "__main__":
    pop_size = 4
    species_count = 2
    random.seed(5)
    # cppn_database = CPPN.CPPN_Database(pop_size, 6, 1, number_of_species=5)
    in_count = 25
    out_count = 4
    # queue = Queue()
    num_gens = 10000
    # Process(target = testing.walker, args = (queue,)).start()
    substrate = Substrate(10, 25, in_count, out_count, CPPN.ACTIVATION_FUNCTIONS["sigmoid"],
                          [
                              #foraging_task.test_agent
			      #snakeProblem.runGameAverag
                              testing.bipedal_walker_no_render_hard_distance_reward,
                              #testing.bipedal_walker_no_render_hard_in_built
                              #t_maze.evaluate_agent
                              #testing.bipedal_walker_no_render_easy_distance
                          ]
                          , pop_size, species_count, num_gens, plasticity=True, diffusion_gradients=2)
    substrate.run()
    # Process(target = testing.bipedal_walker, args = (substrate[fitnesses.index(max(fitnesses))], True)).start()
    # for i in my_linspace(-10, 10, 100):
    # print(cppn_database[0].run([i, -i]))
    #    print(cppn_database[fitnesses.index(max(fitnesses))].run([i, i, i, i, i, i]))
    while input() == "":
        # snakeProblem.displayStrategyRun(substrate[fitnesses.index(max(fitnesses))])
        # snakeProblem.displayStrategyRun(substrate[random.randint(0,pop_size)])
        hard = bool(substrate.choice % 2)
        distance = bool(substrate.choice / 2 % 1)
        testing.bipedal_walker(substrate.get_best_ind(), True, hard, distance)
