from neat import activations
import random
import numpy
import sys
from copy import deepcopy as dc
# from PIL import Image
# from PIL import ImageTk as itk
# import tkinter
# from sklearn.metrics import mean_squared_error as mse
from dsp_rule import Dsp_rule
import math
import function_set

ACTIVATION_FUNCTIONS = activations.ActivationFunctionSet().functions


def ramp(x): return 1 - 2 * (x % 1)


def step(x):
    if (x - x % 1) % 2 == 0:
        return 1
    else:
        return -1


def spike(x):
    if (x - x % 1) % 2 == 0:
        return 1 - 2 * (x % 1)
    else:
        return -1 + 2 * (x % 1)


def euclidian_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[0]) ** 2)


def get_diffusion_gradient(number_of_gradients, coordinates):
    range = 1 / number_of_gradients
    y_coords = my_linspace(-1 + range, 1 - range, number_of_gradients)
    for y_coord in y_coords:
        if euclidian_distance([0, y_coord], coordinates) < range:
            return True, y_coords.index(y_coord)
    return False, None


def get_random_activation(pattern):
    global ACTIVATION_FUNCTIONS
    # keys = list(ACTIVATION_FUNCTIONS.keys())
    # keys.remove("identity")
    # keys.remove("exp")
    # keys.remove("square")
    # keys.remove("cube")
    # keys.remove("inv")
    keys = [ACTIVATION_FUNCTIONS["sin"], ACTIVATION_FUNCTIONS["tanh"], ACTIVATION_FUNCTIONS["sigmoid"],
            ACTIVATION_FUNCTIONS["gauss"], ACTIVATION_FUNCTIONS["inv"], ACTIVATION_FUNCTIONS["tanh"],
            ramp, step, spike]
    if pattern:
        return random.choice(keys)
    else:
        return ACTIVATION_FUNCTIONS["sigmoid"]


keys = [ramp, step, spike]
for f in ACTIVATION_FUNCTIONS.keys():
    keys.append(ACTIVATION_FUNCTIONS[f])


def my_linspace(bottom, top, iter_count):
    if iter_count == 0:
        return []
    to_return = [bottom]
    iterator = float(top - bottom) / float(iter_count)
    for i in range(iter_count - 1):
        to_return.append(to_return[-1] + iterator)
    return to_return


class CPPN:
    # The structure for this is weird. below is the description:
    # self.nodes is a list of nodes. each node contains [a list of connections, the nodes activation function,
    # the current sum, the number of inputs required for the node to output and the number of inputs it has received.
    # ADDITION 17.09.19, nodes now have an additional unique ID for crossover purposes
    # the connections in the list are each 3 element lists containing the innov number, the weight and the target
    def __init__(self, mother, father, in_count, out_count, new_individual=False, allow_recurrent_connections=False,
                 plasticity=False, dsp_threshold=0, diffusion_gradients=0, id=0, pattern=False,
                 time_series=False, watch_and_learn=False, output_activation="identity",
                 initial_weight_distribution=[0, 1]):
        global ACTIVATION_FUNCTIONS
        self.nodes = []
        self.correct_node_order = None
        self.current_recurrent = 0
        self.recurrent_connections = []
        self.time_series = time_series
        if time_series:
            in_count = in_count + out_count
            self.recurrent_data = [0 for i in range(out_count)]
        self.in_count = in_count
        self.original_in_count = in_count
        self.out_count = out_count
        self.allow_recurrent_connections = True #TODO: fix this so it can be from the network declaration
        self.precalculated_outputs = {}
        self.plasticity = plasticity
        self.dsp_threshold = dsp_threshold
        self.diffusion_gradients = diffusion_gradients
        self.plasticity = plasticity
        self.last_reward = None
        self.id = id
        self.pattern = pattern
        self.output_activation = output_activation
        self.input_ids = []
        if not self.pattern:
            self.rule_child = CPPN(None, None, 3 * self.diffusion_gradients + 1, 1, True, plasticity=True, pattern=True)
        else:
            self.rule_child = None
        self.watch_and_learn = watch_and_learn
        if self.watch_and_learn:
            self.watcher = CPPN(None, None, in_count + out_count, diffusion_gradients, True,
                                allow_recurrent_connections=False, plasticity=True, time_series=True)
        if new_individual:
            self.unique_id_to_index = {index: index for index in range(self.in_count + self.out_count)}
            input_coords = my_linspace(-1, 1, in_count)
            for i in range(in_count):
                self.nodes.append([
                    [[i * out_count + target - in_count,
                      random.gauss(initial_weight_distribution[0], initial_weight_distribution[1]), target, None] for
                     target in range(in_count, in_count + out_count)],
                    ACTIVATION_FUNCTIONS["identity"], 0, [0, {}], 0, i,
                    [random.gauss(0, 1) for i in range(self.diffusion_gradients)]])
                self.input_ids.append(i)
            output_coords = my_linspace(-1, 1, out_count)
            for i in range(out_count):
                self.nodes.append(
                    [[], ACTIVATION_FUNCTIONS[self.output_activation], 0,
                     [in_count, {out_count * in_id + i: None for in_id in range(in_count)}],
                     0, self.in_count + i, [random.gauss(0, 1) for i in range(self.diffusion_gradients)]])
            if plasticity:
                self.dsp_rule = Dsp_rule(None, None, new_individual, dsp_threshold)
                self.eta = 0.5
        else:
            self.unique_id_to_index = {index: index for index in range(self.in_count + self.out_count)}
            input_coords = my_linspace(-1, 1, in_count)
            for i in range(in_count):
                self.nodes.append([
                    [],
                    ACTIVATION_FUNCTIONS["identity"], 0, [0, {}], 0, i, [0 for i in range(self.diffusion_gradients)]
                ]
                )
            output_coords = my_linspace(-1, 1, out_count)
            for i in range(out_count):
                self.nodes.append(
                    [[], ACTIVATION_FUNCTIONS[self.output_activation], 0, [0, {}],
                     0, self.in_count + i, [0 for i in range(self.diffusion_gradients)]])
            m_connections = mother.as_dict()
            f_connections = father.as_dict()
            new_connections = []
            for m_connection in m_connections:
                if m_connection in f_connections:
                    new_connections.append(random.choice([m_connections[m_connection], f_connections[m_connection]]))
                else:
                    new_connections.append(m_connections[m_connection])
            for f_connection in f_connections:
                if f_connection not in m_connections:
                    new_connections.append(f_connections[f_connection])
            # first, populate the node list from the targets in new connections
            for connection in new_connections:
                if connection[2] not in self.unique_id_to_index:
                    if connection[2] in mother:
                        activation = mother[connection[2]][1]
                        if connection[2] in father:
                            coords = list(
                                map(lambda m, f: (m + f) / 2, mother[connection[2]][6], father[connection[2]][6]))
                        else:
                            coords = dc(mother[connection[2]][6])
                    else:
                        activation = father[connection[2]][1]
                        coords = dc(father[connection[2]][6])
                    self.nodes.append([[], activation, 0, [0, {}], 0, connection[2], coords])
                    self.unique_id_to_index[connection[2]] = len(self.nodes) - 1
                self.nodes[self.unique_id_to_index[connection[2]]][3][0] += 1
                self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]] = None
            for connection in new_connections:
                try:
                    self.nodes[self.unique_id_to_index[connection[3]]][0].append(connection[:3])
                except KeyError:
                    print(self.unique_id_to_index)
                    print(connection)
                    for node in self.nodes:
                        print(node)
                    raise KeyError()
            if plasticity:
                self.dsp_rule = Dsp_rule(mother.dsp_rule, father.dsp_rule, new_individual, dsp_threshold)
                self.eta = (mother.eta + father.eta) / 2

        self.nat_data = [[] for node in self.nodes]
        if self.plasticity:
            self.hebbs = [[0 for i in self.nodes] for j in self.nodes]
            self.alphas = [[0.1 for i in self.nodes] for j in self.nodes]
        self.mutation_log = ""

    def connection_is_recurrent(self, connection):
        return self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]] is not None

    def is_node_dependent_on_self(self, origin_node_id, current_node_id=None, nodes_visited=[]):
        if current_node_id is None:
            current_node_id = origin_node_id
            nodes_visited = []
        else:
            nodes_visited += [current_node_id]
        for connection in self.nodes[self.unique_id_to_index[current_node_id]][0]:
            if self.allow_recurrent_connections and self.connection_is_recurrent(connection):
                continue
            if connection[2] in range(self.in_count, self.in_count + self.out_count):
                continue
            if connection[2] == origin_node_id:
                return [True, connection]
            if connection[2] in nodes_visited:
                continue
            temp = self.is_node_dependent_on_self(origin_node_id, current_node_id=connection[2],
                                                  nodes_visited=nodes_visited)
            if temp[0]:
                return temp + [connection]
        return False,

    def get_runnable(self):
        for node in self.nodes:
            node[4] = 0
            for connection in node[3][1]:
                node[3][1][connection] = None
            node[3][0] = len(node[3][1])
        if self.plasticity:
            self.hebbs = [[0 for i in self.nodes] for j in self.nodes]
            self.alphas = [[0.1 for i in self.nodes] for j in self.nodes]
        for node in self.nodes:
            node_sum = sum(map(lambda x: x[0] ** 2, node[0]))
            if node_sum != 0:
                for connection in node[0]:
                    connection[1] /= math.sqrt(node_sum)
        self.correct_node_order = []
        self.current_recurrent = 0
        finished = False
        while not finished:
            node_added = False
            for node in range(len(self.nodes)):
                if self.nodes[node][3][0] <= self.nodes[node][4] and node not in self.correct_node_order:
                    node_added = True
                    self.correct_node_order.append(node)
                    for connection in self.nodes[node][0]:
                        self.nodes[self.unique_id_to_index[connection[2]]][4] += 1
            if not node_added:
                # this is where it gets whack. This code needs to find loops within the network and then decide which
                # connection in said loop is recursive. The connection which was most recently added to the network is
                # the one which should be recursive. This connection is the one with the highest innovation number
                # first, find all the nodes dependent on their own output. ignore nodes dependent on nodes which are
                # dependent on another node which is dependent on itself
                found_dependency = False
                for node in range(len(self.nodes)):
                    node_uid = self.index_to_unique(node)
                    if node not in self.correct_node_order and self.nodes[node][3][0] != self.nodes[node][4]:
                        temp = self.is_node_dependent_on_self(node_uid)
                        if temp[0]:
                            found_dependency = True
                            connection_with_highest_innov = temp[1]
                            for connection in temp[2:]:
                                if connection[0] > connection_with_highest_innov[0]:
                                    connection_with_highest_innov = connection
                            target_node = self.nodes[self.unique_id_to_index[connection_with_highest_innov[2]]]
                            target_node[3][0] -= 1
                            target_node[3][1][connection_with_highest_innov[0]] = self.current_recurrent
                            self.current_recurrent += 1
                if not found_dependency:
                    self.print_graph_data()
                    raise RuntimeError("Failed to find any dependencies")

            if len(self.correct_node_order) == len(self.nodes):
                finished = True
        self.recurrent_connections = [0 for i in range(self.current_recurrent)]
        self.nat_data = [[] for node in self.nodes]

    def run(self, inputs):
        for node in self.nodes:
            node[2] = 0
        if self.time_series:
            additional_input = self.recurrent_data
        else:
            additional_input = []
        input_length = len(inputs) + len(additional_input)
        if input_length != self.in_count - 1:
            raise ValueError("This is not the correct amount of inputs, I received " + str(len(inputs)) +
                             " but i expected " + str(self.in_count - 1))
        inputs = inputs[:self.original_in_count] + [1] + additional_input + inputs[self.original_in_count:]
        counter = 0
        for index in list(map(lambda x: self.unique_id_to_index[x], self.input_ids)):
            self.nodes[index][2] = inputs[counter]
            counter += 1
        if self.correct_node_order is None:
            self.get_runnable()
        for node in self.correct_node_order:
            for input_arc in self.nodes[node][3][1]:
                if self.nodes[node][3][1][input_arc] is not None:
                    self.nodes[node][2] += self.recurrent_connections[self.nodes[node][3][1][input_arc]]
        for node in self.correct_node_order:
            self.nodes[node][2] = self.nodes[node][1](self.nodes[node][2])
            if self.plasticity:
                self.nat_data[node].append(0 if self.nodes[node][2] < 0 else 1)
            for connection in self.nodes[node][0]:
                try:
                    if self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]] is None:
                        self.nodes[self.unique_id_to_index[connection[2]]][2] += \
                            self.nodes[node][2] * (
                                    connection[1] + self.alphas[node][self.unique_id_to_index[connection[2]]] *
                                    self.hebbs[node][self.unique_id_to_index[connection[2]]])
                    else:
                        self.recurrent_connections[
                            self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]]] = self.nodes[node][
                                                                                                           2] * \
                                                                                                       connection[1]
                except KeyError as err:
                    print("aaaaaaaa")
                    raise err
        outputs = list(map(lambda x: x[2], self.nodes[self.original_in_count: self.original_in_count + self.out_count]))
        if self.pattern:
            outputs = list(map(lambda x: 0 if -0.2 < x < 0.2 else x, outputs))
            outputs = list(map(lambda x: -1 if x < -1 else x, outputs))
            outputs = list(map(lambda x: 1 if x > 1 else x, outputs))
            # outputs = list(map(lambda x: x * 3, outputs))
        # if self.plasticity:
        #    self.hebbian_plasticity()
        if self.time_series:
            self.recurrent_data = dc(outputs)
        if self.watch_and_learn:
            self.watcher.run(inputs + outputs)
        return outputs

    def _clip(self, val):
        val = -1 if val < -1 else (1 if val > 1 else val)
        return val

    def hebbian_plasticity(self):
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                self.hebbs[i][j] = self._clip(self.hebbs[i][j] + self.eta * self.nodes[i][2] * self.nodes[j][2])

    def revert_changes(self):
        self.hebbs = [[0 for i in self.nodes] for j in self.nodes]

    def keep_changes(self):
        for node in range(len(self.nodes)):
            for connection in self.nodes[node][0]:
                connection[1] += self.alphas[node][self.unique_id_to_index[connection[2]]] * self.hebbs[node][
                    self.unique_id_to_index[connection[2]]]

    def learn(self, reward_signals):
        if self.watch_and_learn:
            reward_signals = dc(self.watcher.recurrent_data)
            self.watcher.recurrent_data = [0 for i in range(self.diffusion_gradients)]
        if len(reward_signals) != self.diffusion_gradients:
            raise ValueError(str(reward_signals) + ", which is of length " + str(
                len(reward_signals)) + " was passed in to the network, we expected " +
                             str(self.diffusion_gradients) + " signals.")
        for node in range(len(self.nodes)):
            for connection in self.nodes[node][0]:
                # nat = [0 for i in range(4)]
                # for line in range(len(self.nat_data[0])):
                #    nat[self.nat_data[node][line] * 2 + self.nat_data[self.unique_id_to_index[connection[2]]][line]] += 1
                if connection[3] is None:
                    connection[3] = connection[1]
                connection[1] += 0.1 * self.rule_child.run(
                    self.nodes[node][6] + self.nodes[self.unique_id_to_index[connection[2]]][6] + reward_signals)[0]
            node_sum = sum(map(lambda x: x[0] ** 2, self.nodes[node][0]))
            for connection in self.nodes[node][0]:
                connection[1] /= math.sqrt(node_sum)
        self.nat_data = [[] for node in self.nodes]

    def revert(self):
        for node in self.nodes:
            for connection in node[0]:
                if connection[3] is None:
                    raise RuntimeError(
                        "The old weight should have been saved. Either revert is being called on a network that hasn't done any learning or something is wrong with this object (whoops)")
                else:
                    connection[1] = connection[3]
                    connection[3] = None

    def print_graph_data(self):
        for node in range(len(self.nodes)):
            print(self.index_to_unique(node))
        for node in range(len(self.nodes)):
            for connection in self.nodes[node][0]:
                print(str(node) + " " + str(connection[2]))

    def get_not_output(self):
        return list(range(self.original_in_count)) + list(
            range(self.original_in_count + self.out_count, len(self.nodes)))

    def mut_add_node(self, innov, node_unique_id):
        connections = []
        for node in range(len(self.nodes)):
            connections.extend([[node, connection] for connection in self.nodes[node][0]])
        chosen_arc = random.choice(connections)
        self.nodes.append(
            [[[innov + 1, 1, chosen_arc[1][2], None]], get_random_activation(self.pattern), 0, [1, {innov: None}], 0,
             node_unique_id, list(map(lambda x, y: random.choice([x, y]),
                                      self.nodes[self.unique_id_to_index[chosen_arc[1][2]]][6],
                                      self.nodes[chosen_arc[0]][6]))
             ])
        self.unique_id_to_index[node_unique_id] = len(self.nodes) - 1
        self.nodes[self.unique_id_to_index[chosen_arc[1][2]]][3][1].pop(chosen_arc[1][0])
        self.nodes[self.unique_id_to_index[chosen_arc[1][2]]][3][1][innov + 1] = None
        chosen_arc[1][2] = node_unique_id
        chosen_arc[1][0] = innov
        self.correct_node_order = None
        self.precalculated_outputs = {}
        self.mutation_log += "mut_add_node, "

    def mut_node_coordinates(self):
        node = random.choice(self.nodes)
        vector = [random.gauss(0, 0.1) for i in range(self.diffusion_gradients)]
        node[6] = list(map(lambda x, y: x + y, node[6], vector))
        self.mutation_log += "mut_node_coordinates, "

    def mut_add_connection(self, innov):
        # find list of possible sources
        self.mutation_log += "mut_add_connection, "
        source_list = []
        for node in self.get_not_output():
            if len(self.nodes[node][0]) < len(self.nodes) - self.in_count:
                source_list.append(node)
        if len(source_list) == 0:
            return
        source_node = random.choice(source_list)
        possible_connections = list(filter(lambda x: x not in self.input_ids, list(self.unique_id_to_index.keys())))
        source_node = self.nodes[source_node]
        for connection in source_node[0]:
            try:
                possible_connections.remove(connection[2])
            except ValueError:
                continue
        new_target = random.choice(possible_connections)
        source_node[0].append([innov, random.gauss(0, 0.1), new_target, None])
        self.nodes[self.unique_id_to_index[new_target]][3][0] += 1
        self.nodes[self.unique_id_to_index[new_target]][3][1][innov] = None
        if not self.allow_recurrent_connections and self.is_node_dependent_on_self(new_target):
            self.nodes[self.unique_id_to_index[new_target]][3][0] -= 1
            self.nodes[self.unique_id_to_index[new_target]][3][1].pop(innov)
            source_node[0].pop()
            return
        self.correct_node_order = None
        self.precalculated_outputs = {}

    def mut_weight(self):
        for node in self.get_not_output():
            for connection in self.nodes[node][0]:
                if random.random() < 0.05:
                    connection[1] += random.gauss(0, 0.1)
        self.precalculated_outputs = {}
        self.mutation_log += "mut_weight, "

    def add_inputs(self, new_input_count, innov_number):
        for i in range(new_input_count):
            innovs = [innov_number + target - self.original_in_count for target in
                      range(self.original_in_count, self.original_in_count + self.out_count)]
            self.nodes.append([[], ACTIVATION_FUNCTIONS["identity"], 0, [0, {}], 0, len(self.nodes) + i,
                               [random.gauss(0, 1) for j in range(self.diffusion_gradients)]])
            for out_node in range(self.original_in_count, self.original_in_count + self.out_count):
                self.nodes[-1][0].append(
                    [innovs[out_node - self.original_in_count], random.gauss(0, 0.1), out_node, None])
                self.nodes[out_node][3][0] += 1
                self.nodes[out_node][3][1][innovs[out_node - self.original_in_count]] = None
            innov_number += self.out_count
            self.unique_id_to_index[len(self.nodes) - 1] = len(self.nodes) - 1
            self.input_ids.append(len(self.nodes) - 1)
        self.in_count += new_input_count
        self.get_runnable()
        self.mutation_log += "add_inputs, "

    def index_to_unique(self, index):
        for unique_id in self.unique_id_to_index:
            if self.unique_id_to_index[unique_id] == index:
                return unique_id
        raise RuntimeError("There is a node which is not in the unique id conversion dictionary")

    def as_dict(self):
        dict_of_arcs = {}
        for node in self.get_not_output():
            for connection in self.nodes[node][0]:
                dict_of_arcs[connection[0]] = connection + [self.index_to_unique(node)]
        return dc(dict_of_arcs)

    def __getitem__(self, index):
        return self.nodes[self.unique_id_to_index[index]]

    def __contains__(self, item):
        return item in self.unique_id_to_index

    def __repr__(self):
        string = "CPPN_copy("
        string += "["
        for node in self.nodes:
            node_str = str(node)
            string += node_str[:node_str.index("<")]
            string += "function_set." + node_str[node_str.index("<"):].split(" ")[1]
            string += node_str[node_str.index(">") + 1:]
            string += ", "
        string += "], "
        string += str(self.unique_id_to_index)
        string += ", "
        string += repr(self.rule_child)
        string += ", "
        string += str(self.in_count)
        string += ", "
        string += str(self.original_in_count)
        string += ", "
        string += str(self.out_count)
        string += ", "
        string += str(self.allow_recurrent_connections)
        string += ", "
        string += str(self.plasticity)
        string += ", "
        string += str(self.dsp_threshold)
        string += ", "
        string += str(self.diffusion_gradients)
        string += ", "
        string += str(self.id)
        string += ", "
        string += str(self.time_series)
        string += ", "
        string += str(self.input_ids)
        string += ", "
        string += str(self.pattern)
        string += ")"
        return string


class CPPN_copy(CPPN):
    def __init__(self, nodes, unique_id_to_index, rule_child, in_count, original_in_count, out_count,
                 allow_recurrent_connections=False, plasticity=False, dsp_threshold=0, diffusion_gradients=0,
                 id=0, time_series=False, input_ids=None, pattern=False):
        self.nodes = nodes
        self.correct_node_order = None
        self.current_recurrent = 0
        self.recurrent_connections = []
        self.in_count = in_count
        self.original_in_count = original_in_count
        self.out_count = out_count
        self.allow_recurrent_connections = allow_recurrent_connections
        self.precalculated_outputs = {}
        self.plasticity = plasticity
        self.dsp_threshold = dsp_threshold
        self.diffusion_gradients = diffusion_gradients
        self.plasticity = plasticity
        self.last_reward = None
        self.id = id
        self.pattern = pattern
        self.unique_id_to_index = unique_id_to_index
        self.rule_child = rule_child
        self.time_series = time_series
        if input_ids is None:
            self.input_ids = [i for i in range(in_count)]
        else:
            self.input_ids = input_ids


def tournament_select(population, fitnesses, tournament_size):
    tournament = random.choices(range(len(population)), k=tournament_size)
    best = tournament[0]
    for individual in tournament[1:]:
        if fitnesses[individual] > fitnesses[best]:
            best = individual
    return dc(population[best])


class Species:
    def __init__(self, base_network, id, selection_method, additional_selection_params, in_count, out_count):
        self.base_network = base_network
        self.cppns = [base_network]
        self.id = id
        self.fitnesses = []
        self.innov_number = None
        self.node_unique_id = None
        self.selection_method = selection_method
        self.additional_selection_params = additional_selection_params
        self.in_count = in_count
        self.out_count = out_count
        self.gen = self.time = self.choice = 0

    def distribute_meta(self, gen, time, choice):
        self.gen = gen
        self.time = time
        self.choice = choice

    def get_base(self):
        return self.base_network

    def append(self, network):
        self.cppns.append(network)

    def __iter__(self):
        return self.cppns

    def __getitem__(self, index):
        return self.cppns[index]

    def __len__(self):
        return len(self.cppns)

    def __str__(self):
        statistics = [self.gen, self.id, numpy.std(self.fitnesses), numpy.max(self.fitnesses),
                      numpy.mean(self.fitnesses), len(self.cppns),
                      numpy.min(self.fitnesses), self.time, self.choice]
        string = str(self.gen)
        counter = 0
        for key in statistics[1:]:
            if counter == 0:
                string += "              " + str(round(key, 3))
            else:
                string += "        " + str(round(key, 3))
            if counter < 7:
                if len(str(round(key))) < 4:
                    for i in range(len(str(round(key, 3))), 6):
                        string += " "
            counter += 1
        return string

    def __repr__(self):
        statistics = [self.gen, self.id, numpy.std(self.fitnesses), numpy.max(self.fitnesses),
                      numpy.mean(self.fitnesses), len(self.cppns),
                      numpy.min(self.fitnesses), self.time, self.choice]
        return str(statistics)[1:-1]

    def kill_pop(self):
        self.cppns = []

    def get_networks(self):
        return self.cppns

    def get_best_fitness(self):
        return max(self.fitnesses), self.cppns[self.fitnesses.index(max(self.fitnesses))]

    def mut_add_node(self, target=None):
        if target is None:
            random.choice(self.cppns).mut_add_node(self.innov_number, self.node_unique_id)
        else:
            self.cppns[target].mut_add_node(self.innov_number, self.node_unique_id)
        self.innov_number += 2
        self.node_unique_id += 1

    def mut_add_connection(self, target=None):
        if target is None:
            random.choice(self.cppns).mut_add_connection(self.innov_number)
        else:
            self.cppns[target].mut_add_connection(self.innov_number)
        self.innov_number += 1

    def mut_random_weight(self, target=None):
        if target is None:
            random.choice(self.cppns).mut_weight()
        else:
            self.cppns[target].mut_weight()

    def mutate_population(self, innov_number, node_unique_id):
        self.innov_number, self.node_unique_id = innov_number, node_unique_id
        for cppn in self.cppns:
            if random.random() < 0.99:
                # self.cppns.append(dc(cppn))
                mut_choice = random.randint(0, 2)
                if mut_choice == 0:
                    cppn.mut_add_node(self.innov_number, self.node_unique_id)
                    self.innov_number += 2
                    self.node_unique_id += 1
                elif mut_choice == 1:
                    cppn.mut_add_connection(self.innov_number)
                    self.innov_number += 1
                elif mut_choice == 2:
                    cppn.mut_weight()
        # for i in range(random.randint(0, round(len(self.cppns) / 5))):
        #    self.mut_add_node()
        # for i in range(random.randint(0, round(len(self.cppns) / 5))):
        #    self.mut_add_connection()
        # for i in range(random.randint(0, round(len(self.cppns) / 5))):
        #    self.mut_random_weight()
        return self.innov_number, self.node_unique_id

    def select_n_individuals(self, n, hof=0):
        new_pop = []
        fitnesses_copy = dc(self.fitnesses)
        fitnesses_copy.sort()
        for individual in range(min(len(self.cppns), hof)):
            new_pop.append(dc(self.cppns[self.fitnesses.index(fitnesses_copy[individual])]))
        for individual in range(n - hof):
            parents = [self.selection_method(self.cppns, self.fitnesses, *self.additional_selection_params) for i in
                       range(2)]
            # new_pop.append(CPPN(*parents, self.in_count, self.out_count))
            new_pop.append(CPPN(parents[0], parents[1], self.in_count, self.out_count,
                                plasticity=parents[0].plasticity, dsp_threshold=parents[0].dsp_threshold,
                                diffusion_gradients=parents[0].diffusion_gradients))
        return new_pop

    #    def save_best_fitness(self, filename, gen_no):
    #        self.get_best_fitness()[1].save_to_file(filename, gen_no)

    def __lt__(self, other_species):
        return self.get_best_fitness()[0] < other_species.get_best_fitness()[0]


class CPPN_Database:
    def __init__(self, pop_size, in_count, out_count, hof=2, selection_method=tournament_select, \
                 additional_selection_params=[3], number_of_species=0, plasticity=False, dsp_threshold=0,
                 diffusion_gradients=0):
        self.in_count = in_count + 1
        self.out_count = out_count
        self.pop_size = pop_size
        self.innov_number = self.in_count * self.out_count
        self.node_unique_id = self.in_count + self.out_count + 1
        self.hof = hof
        self.selection_method = selection_method
        self.additional_selection_params = additional_selection_params
        self.c1 = 0.03
        self.c2 = 0.03
        self.number_of_species = number_of_species
        self.plasticity, self.dsp_threshold, self.diffusion_gradients = plasticity, dsp_threshold, diffusion_gradients
        cppns = [CPPN(None, None, self.in_count, self.out_count, True, plasticity=plasticity, dsp_threshold=
        dsp_threshold, diffusion_gradients=diffusion_gradients) for individual in range(self.pop_size)]
        self.species = [
            Species(cppns[0], 0, selection_method, additional_selection_params, self.in_count, self.out_count)]
        self.species_id = 1
        for cppn in cppns:
            self.add_to_species(cppn)

    def save_to_file(self):
        with open("output_networks.ann", "a") as output_file:
            output_file.write(repr(self.species[0].get_best_fitness()[1]))

    def mutate_population(self):
        for species in self.species:
            self.innov_number, self.node_unique_id = species.mutate_population(self.innov_number, self.node_unique_id)

    def distribute_meta(self, gen, time, choice):
        for species in self.species:
            species.distribute_meta(gen, time, choice)

    def new_generation(self, fitnesses):
        # self.distribute_fitnesses(fitnesses)
        self.species.sort(reverse=True)
        self.species = self.species[:self.number_of_species]
        protected = [i[0] for i in self.species]
        new_pop = []
        for ind in protected:
            new_pop.extend(dc(ind) for i in range(5))
        # for species in self.species:
        #    tmp = species.select_n_individuals(self.pop_size, hof = self.hof)
        #    new_pop.extend(tmp[self.hof:])
        #    protected.extend(tmp[:self.hof])
        #    species.kill_pop()

        for individual in new_pop:
            self.add_to_species(individual)
        self.mutate_population()
        for individual in protected:
            self.add_to_species(individual)
        remove_count = 0
        for species in range(len(self.species)):
            if len(self.species[species]) == 0:
                self.species[species] = None
                remove_count += 1
        for iterator in range(remove_count):
            self.species.remove(None)

    def distribute_fitnesses(self, fitnesses):
        iterator = 0
        for species in self.species:
            species.fitnesses = fitnesses[iterator:iterator + len(species)]
            iterator += len(species)

    def add_to_species(self, cppn):
        differences = []
        for species in self.species:
            differences.append(self._get_speciation_difference(species.get_base(), cppn))
        if min(differences) >= 0:  # this is a hyperparameter, but can remain constant while c1 and c2 are changed
            self.species.append(
                Species(cppn, self.species_id, self.selection_method, self.additional_selection_params, self.in_count,
                        self.out_count))
            self.species_id += 1
        else:
            self.species[differences.index(min(differences))].append(cppn)

    def _get_speciation_difference(self, network1, network2):
        arcs = (network1.as_dict(), network2.as_dict())
        counts = [0, 0]  # excess / disjoint, sum of matching innov weight difference
        for innov in range(self.innov_number):
            if innov in arcs[0] and innov in arcs[1]:
                counts[1] += abs(arcs[0][innov][1] - arcs[1][innov][1])
            elif innov in arcs[0] or innov in arcs[1]:
                counts[0] += 1
        return counts[0] * self.c1 + counts[1] * self.c2

    def get_cppns(self):
        cppns = []
        for species in self.species:
            cppns.extend(species.cppns)
        return cppns

    def __str__(self):
        string = ""
        self.species.sort(reverse=True)
        for species in self.species:
            string += str(species) + "\n"
        return string

    def __repr__(self):
        string = ""
        self.species.sort(reverse=True)
        for species in self.species:
            string += repr(species) + "\n"
        return string


def display_image(cppn):
    image = []
    for x in range(256):
        image.append([])
        for y in range(256):
            x_tmp = float(x) / 128
            x_tmp = x_tmp - 1
            y_tmp = float(y) / 128
            y_tmp = y_tmp - 1
            image[-1].append(numpy.array(cppn.run([x_tmp, y_tmp]))[0] * 256)
            for rgb in range(1):
                if image[-1][-1] < 0:
                    image[-1][-1] = 0
                if image[-1][-1] > 255:
                    image[-1][-1] = 255
    #            if image[-1][-1] < 0:
    #                image[-1][-1] = 0
    #            elif image[-1][-1] > 255:
    #                image[-1][-1] = 255
    image = numpy.asarray(image, dtype='uint32')
    print(image)
    tmp = Image.fromarray(image, mode="I")
    tmp.show()


def mutate_individual(innov_number, testing_function, cppn, mutation_target = None):
    # self.cppns.append(dc(cppn))
    if type(cppn) == list:
        out = [mutate_individual(innov_number, testing_function, cppn_i, mutation_target) for cppn_i in cppn]
        return out
    elif cppn.pattern:
        mut_choice = random.randint(0, 3)
        if mut_choice == 0:
            cppn.mut_add_node(innov_number, len(cppn.nodes))
        elif mut_choice == 1:
            cppn.mut_add_connection(innov_number)
        elif mut_choice == 2:
            cppn.mut_weight()
    elif mutation_target is None:
        mut_choice = random.randint(0, 4 + int(cppn.watch_and_learn))
        if mut_choice == 0:
            cppn.mut_add_node(innov_number, len(cppn.nodes))
        elif mut_choice == 1:
            cppn.mut_add_connection(innov_number)
        elif mut_choice == 2:
            cppn.mut_weight()
        elif mut_choice == 3:
            cppn.mut_node_coordinates()
        elif mut_choice == 4:
            mutate_individual(innov_number, None, cppn.rule_child)
        if cppn.watch_and_learn and mut_choice == 5:
            mutate_individual(innov_number, None, cppn.watcher)
            # fitnesses.append(testing_function(cppns[i]))
    elif mutation_target == 0:
        mut_choice = random.randint(0, 3)
        if mut_choice == 0:
            cppn.mut_add_node(innov_number, len(cppn.nodes))
        elif mut_choice == 1:
            cppn.mut_add_connection(innov_number)
        elif mut_choice == 2:
            cppn.mut_weight()
        elif mut_choice == 3:
            cppn.mut_node_coordinates()
    elif mutation_target == 1:
        mutate_individual(innov_number, None, cppn.rule_child)
    else:
        mutate_individual(innov_number, None, cppn.watcher)
    return cppn


if __name__ == "__main__":
    cppn = CPPN(None, None, 9, 5, True, True, True, 0, 100, id=0)
    # cppn_i = eval(repr(cppn))
    # file = open("final_layer_log.log", "r")
    # for line in file:
    #    cppn = eval(line)
    print(cppn.run([1 for i in range(cppn.in_count - 1)]))
    for i in range(100):
        if random.randint(0, 1):
            mutate_individual(i, None, cppn)
        else:
            cppn.add_inputs(9, i)
    print(cppn.run([1 for i in range(cppn.in_count - 1)]))
    # print(cppn_i.run([i for i in range(9)]))

if __name__ == "herbert":
    in_count = 2
    out_count = 1
    cppn_database = CPPN_Database(20, in_count, out_count, number_of_species=7)
    image = Image.open("ewe.jpg")
    image = image.resize([128, 128])
    image = image.convert('LA')
    x_range = y_range = 5
    root = tkinter.Tk()
    root.geometry("500x500")
    boolean = True
    for i in range(500):
        if i > 0:
            cppn_database.distribute_fitnesses(dc(fitnesses))
            cppn_database.new_generation(dc(fitnesses))
        cppn_list = cppn_database.get_cppns()
        fitnesses = [0 for k in range(len(cppn_list))]
        for cppn in range(len(cppn_list)):
            corrects = []
            found = []
            for x in random.sample(range(0, 127), x_range):
                tmp = []
                for y in random.sample(range(0, 127), y_range):
                    coords = (x, y)
                    correct_value = numpy.array(image.getpixel(coords)) / 255
                    corrects.append(correct_value[0])
                    # correct_value = 0.5
                    coords = ((coords[0] / 64) - 1, (coords[1] / 64) - 1)
                    output = cppn_list[cppn].run(coords)
                    for out in range(len(output)):
                        if output[out] > 1:
                            output[out] = 1
                        elif output[out] < 0:
                            output[out] = 0
                    found.append(output[0])
                    # print(str(correct_value) + str(output) + str(mse(correct_value, output)))
                    # fitnesses[cppn] += mse(correct_value, output)
                    # fitnesses[cppn] = (correct_value[0] - output[0]) ** 2
                    # tmp.append(sum(output))
                    # if sum(output) > correct_value:
                    #    fitnesses[cppn] -= (sum(output) - correct_value)
                    # else:
                    #    fitnesses[cppn] -= (correct_value - sum(output))
            # fitnesses[cppn] = numpy.var(found)
            fitnesses[cppn] = - mse(corrects, found)
            # fitnesses[cppn] /= (x_range * y_range)
            # fitnesses[cppn] = (1 / fitnesses[cppn])

        print(str(i) + "      " + str(numpy.mean(fitnesses)))
        # print(cppn_database[fitnesses.index(max(fitnesses))].run([0,0,0,0,0,0]))
        # print(cppn_database[fitnesses.index(max(fitnesses))].run([10,10,10,10,10,10]))
        # print(len(fitnesses))
        image_tmp = []
        for x in range(256):
            image_tmp.append([])
            for y in range(256):
                x_tmp = float(x) / 128
                x_tmp = x_tmp - 1
                y_tmp = float(y) / 128
                y_tmp = y_tmp - 1
                image_tmp[-1].append(
                    numpy.array(cppn_list[fitnesses.index(max(fitnesses))].run([x_tmp, y_tmp]))[0] * 256)
                for rgb in range(1):
                    if image_tmp[-1][-1] < 0:
                        image_tmp[-1][-1] = 0
                    if image_tmp[-1][-1] > 255:
                        image_tmp[-1][-1] = 255
        #            if image_tmp[-1][-1] < 0:
        #                image_tmp[-1][-1] = 0
        #            elif image_tmp[-1][-1] > 255:
        #                image_tmp[-1][-1] = 255
        image_tmp = numpy.asarray(image_tmp, dtype='uint32')
        # print(image_tmp)
        tmp = Image.fromarray(image_tmp, mode="I")
        x = itk.PhotoImage(tmp)
        if boolean:
            tkLabel = tkinter.Label(root, image=x)
            boolean = False
        else:
            tkLabel.configure(image=x)
        # tkLabel.place(x=0, y=-1000000)
        # tkLabel.photo = x
        tkLabel.pack()
        root.update()
        # print(type(tmp))
        # tmp.show()
