import random
from copy import deepcopy as dc
import math
import numpy as np
import numba
from numba import cuda


def relu(x):
    if x < 0:
        return 0
    else:
        return x

def identity(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

ACTIVATION_FUNCTIONS = {"relu": relu, "identity": identity, "sigmoid":sigmoid}

def my_linspace(bottom, top, iter_count):
    if iter_count == 0:
        return []
    to_return = [bottom]
    iterator = float(top - bottom) / float(iter_count)
    for i in range(iter_count - 1):
        to_return.append(to_return[-1] + iterator)
    return to_return


class Network:
    def __init__(self, in_count, out_count, allow_recurrent_connections, has_child=True, diffusion_gradients=0,
                 watch_and_learn=False, time_series=False, output_activation= "identity"):
        self.nodes = []
        self.correct_node_order = None
        self.current_recurrent = 0
        self.recurrent_connections = []
        self.recurrent_data = []
        self.in_count = in_count
        self.original_in_count = in_count
        self.out_count = out_count
        self.allow_recurrent_connections = allow_recurrent_connections
        self.last_reward = None
        self.output_activation = output_activation
        self.input_ids = []
        self.has_child = has_child
        self.time_series = time_series
        self.diffusion_gradients = diffusion_gradients
        self.mutation_log = ""
        if self.has_child:
            self.rule_child = Network(3 * self.diffusion_gradients + 1, 1, True, False)
        else:
            self.rule_child = None
        self.watch_and_learn = watch_and_learn
        if self.watch_and_learn:
            self.watcher = Network(in_count + out_count, diffusion_gradients, False, time_series=True)
        initial_weight_distribution = [0, 0.333]
        for i in range(in_count):
            self.nodes.append([
                [[i * out_count + target - in_count,
                  random.gauss(initial_weight_distribution[0], initial_weight_distribution[1]), target, None] for
                 target in range(in_count, in_count + out_count)],
                ACTIVATION_FUNCTIONS["identity"], 0, [0, {}], 0, i,
                [random.gauss(0, 0.333) for i in range(self.diffusion_gradients)]])
            self.input_ids.append(i)
        for i in range(out_count):
            self.nodes.append(
                [[], ACTIVATION_FUNCTIONS[self.output_activation], 0,
                 [in_count, {out_count * in_id + i: None for in_id in range(in_count)}],
                 0, self.in_count + i, [random.gauss(0, 1) for i in range(self.diffusion_gradients)]])
        self.current_innov = self.in_count * self.out_count

    def connection_is_recurrent(self, connection):
        return self.nodes[connection[2]][3][1][connection[0]] is not None

    def is_node_dependent_on_self(self, origin_node_id, current_node_id=None, nodes_visited=[]):
        if current_node_id is None:
            current_node_id = origin_node_id
            nodes_visited = []
        else:
            nodes_visited += [current_node_id]
        for connection in self.nodes[current_node_id][0]:
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
        #for node in self.nodes:
        #    node_sum = sum(map(lambda x: x[1] ** 2, node[0]))
        #    if node_sum != 0:
        #        for connection in node[0]:
        #            connection[1] /= math.sqrt(node_sum)
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
                        self.nodes[connection[2]][4] += 1
            if not node_added:
                # this is where it gets whack. This code needs to find loops within the network and then decide which
                # connection in said loop is recursive. The connection which was most recently added to the network is
                # the one which should be recursive. This connection is the one with the highest innovation number
                # first, find all the nodes dependent on their own output. ignore nodes dependent on nodes which are
                # dependent on another node which is dependent on itself
                found_dependency = False
                for node in range(len(self.nodes)):
                    node_uid = node
                    if node not in self.correct_node_order and self.nodes[node][3][0] != self.nodes[node][4]:
                        temp = self.is_node_dependent_on_self(node_uid)
                        if temp[0]:
                            found_dependency = True
                            connection_with_highest_innov = temp[1]
                            for connection in temp[2:]:
                                if connection[0] > connection_with_highest_innov[0]:
                                    connection_with_highest_innov = connection
                            target_node = self.nodes[connection_with_highest_innov[2]]
                            target_node[3][0] -= 1
                            target_node[3][1][connection_with_highest_innov[0]] = self.current_recurrent
                            self.current_recurrent += 1
                if (not found_dependency):
                    raise ValueError("This is not a valid network")
            if len(self.correct_node_order) == len(self.nodes):
                finished = True
        self.recurrent_connections = [0 for i in range(self.current_recurrent)]

    def run(self, inputs):
        for node in self.nodes:
            node[2] = 0
        if self.time_series:
            additional_input = self.recurrent_data
        else:
            additional_input = []
        input_length = len(inputs) + len(additional_input)
        #if input_length != self.in_count:
        #    raise ValueError("This is not the correct amount of inputs, I received " + str(len(inputs)) +
        #                     " but i expected " + str(self.in_count - 1))
        #inputs = [1] + inputs[:self.original_in_count] + additional_input + inputs[self.original_in_count:]
        counter = 0
        for index in self.input_ids:
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
            for connection in self.nodes[node][0]:
                if self.nodes[connection[2]][3][1][connection[0]] is None:
                    self.nodes[connection[2]][2] += self.nodes[node][2] * connection[1]
                else:
                    self.recurrent_connections[
                        self.nodes[connection[2]][3][1][connection[0]]] = self.nodes[node][2] * connection[1]
        outputs = list(map(lambda x: x[2], self.nodes[self.original_in_count: self.original_in_count + self.out_count]))
        if self.time_series:
            self.recurrent_data = dc(outputs)
        if self.watch_and_learn:
            self.watcher.run(inputs + outputs)
        return outputs

    def learn(self, reward_signals):
        if self.watch_and_learn:
            reward_signals = dc(self.watcher.recurrent_data)
            self.watcher.recurrent_data = [0 for i in range(self.diffusion_gradients)]
        if len(reward_signals) != self.diffusion_gradients:
            raise ValueError(str(reward_signals) + ", which is of length " + str(
                len(reward_signals)) + " was passed in to the network, we expected " +
                             str(self.diffusion_gradients) + " signals.")
        tmp = np.zeros(self.diffusion_gradients * 3 + 1)
        tmp[self.diffusion_gradients*2:-1] = reward_signals
        tmp[-1] = 1
        for node in range(len(self.nodes)):
            for connection in self.nodes[node][0]:
                if connection[3] is None:
                    connection[3] = connection[1]
                tmp[:self.diffusion_gradients] = self.nodes[node][6]
                tmp[self.diffusion_gradients: 2* self.diffusion_gradients] = self.nodes[connection[2]][6]
                connection[1] += 0.1 * self.rule_child.run(tmp)[0]
            node_sum = sum(map(lambda x: x[0] ** 2, self.nodes[node][0]))
            for connection in self.nodes[node][0]:
                connection[1] /= math.sqrt(node_sum)

    def revert(self):
        for node in self.nodes:
            for connection in node[0]:
                if connection[3] is None:
                    raise RuntimeError(
                        "The old weight should have been saved. Either revert is being called on a network that hasn't done any learning or something is wrong with this object (whoops)")
                else:
                    connection[1] = connection[3]
                    connection[3] = None

    def get_not_output(self):
        return list(range(self.original_in_count)) + list(
            range(self.original_in_count + self.out_count, len(self.nodes)))

    def mut_add_node(self):
        connections = []
        for node in range(len(self.nodes)):
            connections.extend([[node, connection] for connection in self.nodes[node][0]])
        chosen_arc = random.choice(connections)
        diffusion_gradient_values = []
        for dg in range(self.diffusion_gradients):
            diffusion_gradient_values.append((self.nodes[chosen_arc[1][2]][6][dg] + self.nodes[chosen_arc[0]][6][dg]) / 2)
        self.nodes.append(
            [[[self.current_innov + 1, 1, chosen_arc[1][2], None]], ACTIVATION_FUNCTIONS["sigmoid"], 0,
             [1, {self.current_innov: None}], 0, None, diffusion_gradient_values])
        self.nodes[chosen_arc[1][2]][3][1].pop(chosen_arc[1][0])
        self.nodes[chosen_arc[1][2]][3][1][self.current_innov + 1] = None
        chosen_arc[1][2] = len(self.nodes) - 1
        chosen_arc[1][0] = self.current_innov
        self.correct_node_order = None
        self.mutation_log += "mut_add_node, "
        self.current_innov += 2

    def mut_node_coordinates(self):
        node = random.choice(self.nodes)
        vector = [random.gauss(0, 0.1) for i in range(self.diffusion_gradients)]
        node[6] = list(map(lambda x, y: x + y, node[6], vector))
        self.mutation_log += "mut_node_coordinates, "

    def mut_add_connection(self):
        self.mutation_log += "mut_add_connection, "
        source_list = []
        for node in self.get_not_output():
            if len(self.nodes[node][0]) < len(self.nodes) - self.in_count:
                source_list.append(node)
        if len(source_list) == 0:
            return
        source_node = random.choice(source_list)
        possible_connections = list(filter(lambda x: x not in self.input_ids, list(range(len(self.nodes)))))
        source_node = self.nodes[source_node]
        for connection in source_node[0]:
            try:
                possible_connections.remove(connection[2])
            except ValueError:
                continue
        new_target = random.choice(possible_connections)
        source_node[0].append([self.current_innov, random.gauss(0, 0.1), new_target, None])
        self.nodes[new_target][3][0] += 1
        self.nodes[new_target][3][1][self.current_innov] = None
        if self.is_node_dependent_on_self(new_target):
            if self.allow_recurrent_connections:
                self.nodes[new_target][3][1][self.current_innov] = self.current_recurrent
                self.current_recurrent += 1
            else:
                self.nodes[new_target][3][0] -= 1
                self.nodes[new_target][3][1].pop(self.current_innov)
                source_node[0].pop()
                return
        self.current_innov += 1
        self.correct_node_order = None

    def mut_weight(self):
        for node in self.get_not_output():
            for connection in self.nodes[node][0]:
                if random.random() < 0.05:
                    connection[1] += random.gauss(0, 0.1)
        self.mutation_log += "mut_weight, "

    def add_inputs(self, new_input_count):
        for i in range(new_input_count):
            innovs = [self.current_innov + target - self.original_in_count for target in
                      range(self.original_in_count, self.original_in_count + self.out_count)]
            self.nodes.append([[], ACTIVATION_FUNCTIONS["identity"], 0, [0, {}], 0, len(self.nodes) + i,
                               [random.gauss(0, 1) for j in range(self.diffusion_gradients)]])
            for out_node in range(self.original_in_count, self.original_in_count + self.out_count):
                self.nodes[-1][0].append(
                    [innovs[out_node - self.original_in_count], random.gauss(0, 0.3), out_node, None])
                self.nodes[out_node][3][0] += 1
                self.nodes[out_node][3][1][innovs[out_node - self.original_in_count]] = None
            self.current_innov += self.out_count
            self.input_ids.append(len(self.nodes) - 1)
        self.in_count += new_input_count
        self.get_runnable()
        self.mutation_log += "add_inputs, "

    def __repr__(self):
        string = "{"
        string += "nodes: ["
        for node in self.nodes:
            node_str = str(node)
            string += node_str[:node_str.index("<")]
            string += "function_set." + node_str[node_str.index("<"):].split(" ")[1]
            string += node_str[node_str.index(">") + 1:]
            string += ", "
        string += "], rule_child: "
        string += repr(self.rule_child)
        string += ", in_count: "
        string += str(self.in_count)
        string += ", original_in_count: "
        string += str(self.original_in_count)
        string += ", out_count: "
        string += str(self.out_count)
        string += ", allow_recurrent_connections: "
        string += str(self.allow_recurrent_connections)
        string += ", diffusion_gradients: "
        string += str(self.diffusion_gradients)
        string += ", time_series: "
        string += str(self.time_series)
        string += ", in_ids: "
        string += str(self.input_ids)
        string += "}"
        return string

    def get_network_gpu_representation(self):
        if self.correct_node_order is None:
            self.get_runnable()
        input_ids = np.array(self.input_ids, dtype=int)
        commands = []
        connection_weights = []
        connection_targets = []
        sources = []
        for node in self.correct_node_order:
            if node not in input_ids:
                commands.append(True)
                connection_weights.append(0)
                connection_targets.append(0)
                sources.append(node)
            for connection in self.nodes[node][0]:
                commands.append(False)
                connection_weights.append(connection[1])
                connection_targets.append(connection[2])
                sources.append(node)

        # commands, connection_weights, connection_targets, sources, input_ids
        return np.array(commands, dtype=bool), np.array(connection_weights, dtype=float), \
               np.array(connection_targets, dtype=int), np.array(sources, dtype=int), input_ids

#removes a bunch of features, only the basic network can be run here. NO RECURRENCY
@cuda.jit
def run_network_gpu(inputs, commands, connection_weights, connection_targets, sources, command_count, original_in_count,
                    in_count, input_ids, outputs, node_vals_big):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array
    bigPos = tx + ty * bw
    pos = bigPos % node_vals_big.shape[1]
    image = bigPos // node_vals_big.shape[1]
    if pos >= node_vals_big.shape[1] or image >= node_vals_big.shape[0]:
        return
    for index in range(in_count):
        node_vals_big[image][pos][input_ids[index]] = inputs[image][pos][index]
    for c in range(command_count):
        if commands[c]:
            node_vals_big[image][pos][sources[c]] = 1 / (1 + 2.718281828459045 ** (-node_vals_big[image][pos][sources[c]]))
        else:
            node_vals_big[image][pos][connection_targets[c]] += node_vals_big[image][pos][sources[c]] * connection_weights[c]
    outputs[image][pos] = node_vals_big[image][pos][original_in_count]


def mutate_individual(network, ruleChild=False):
    if type(network) == list:
        out = [mutate_individual(network_i) for network_i in network]
        return out
    mut_choice = random.randint(0, 4 + int(network.watch_and_learn))
    if mut_choice == 0:
        network.mut_add_node()
    elif mut_choice == 1:
        network.mut_add_connection()
    elif mut_choice == 2:
        network.mut_weight()
    elif mut_choice == 3:
        network.mut_node_coordinates()
    elif mut_choice == 4 and not ruleChild:
        mutate_individual(network.rule_child, ruleChild = True)
    if network.watch_and_learn and mut_choice == 5 and not ruleChild:
        network.watcher = mutate_individual(network.watcher)
    return network

if __name__ == "__main__":
    for j in range(100):
        network = Network(2, 1, False)
        vals = [[], []]
        vals[0].append(network.run([1])[0])
        network.add_inputs(1)
        vals[1].append(network.run([1,1])[0])
        toPrint = ""
        for i in range(len(vals[0])):
            toPrint += str(vals[0][i]) + "\t" + str(vals[1][i]) + "\n"
        print(toPrint)
