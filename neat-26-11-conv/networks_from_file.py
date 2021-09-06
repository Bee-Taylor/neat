import CPPN
import testing
import random
import sys

class Network:
    # The structure for this is weird. below is the description:
    # self.nodes is a list of nodes. each node contains [a list of connections, the nodes activation function,
    # the current sum, the number of inputs required for the node to output and the number of inputs it has received.
    # ADDITION 17.09.19, nodes now have an additional unique ID for crossover purposes
    # the connections in the list are each 3 element lists containing the innov number, the weight and the target
    def __init__(self, nodes, in_count, out_count, new_individual=False, allow_recurrent_connections=False,
                 plasticity = False, dsp_threshold = 0):
        global ACTIVATION_FUNCTIONS
        self.nodes = nodes[:-1]
        self.unique_id_to_index = nodes[-1]
        self.correct_node_order = None
        self.current_recurrent = 0
        self.recurrent_connections = []
        self.in_count = in_count
        self.out_count = out_count
        self.allow_recurrent_connections = allow_recurrent_connections
        self.precalculated_outputs = {}
        self.plasticity = plasticity
        self.dsp_threshold = dsp_threshold


    def connection_is_recurrent(self, connection):
        return self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]] is not None

    def is_node_dependent_on_self(self, origin_node_id, current_node_id=None, nodes_visited = []):
        if current_node_id is None:
            current_node_id = origin_node_id
            nodes_visited = []
        else:
            nodes_visited += [current_node_id]
        for connection in self.nodes[self.unique_id_to_index[current_node_id]][0]:
            if self.connection_is_recurrent(connection):
                continue
            if connection[2] in range(self.in_count, self.in_count + self.out_count):
                continue
            if connection[2] == origin_node_id:
                return [True, connection]
            if connection[2] in nodes_visited:
                continue
            temp = self.is_node_dependent_on_self(origin_node_id, current_node_id=connection[2], nodes_visited = nodes_visited)
            if temp[0]:
                return temp + [connection]
        return False,

    def get_runnable(self):
        for node in self.nodes:
            node[4] = 0
            for connection in node[3][1]:
                node[3][1][connection] = None
            node[3][0] = len(node[3][1])
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

    def run(self, inputs):
        for node in self.nodes:
            node[2] = 0
        if len(inputs) != self.in_count - 1:
            raise ValueError("This is not the correct amount of inputs" + str(len(inputs)) + str(self.in_count))
        for i in range(self.in_count - 1):
            self.nodes[i][2] = inputs[i]
        tmp = self.precalculated_outputs
        for input_i in inputs:
            if input_i in tmp:
                tmp = tmp[input_i]
            else:
                break
        if type(tmp) == list:
            return tmp
        self.nodes[self.in_count - 1][2] = 1
        if self.correct_node_order is None:
            self.get_runnable()
        for node in self.correct_node_order:
            for input_arc in self.nodes[node][3][1]:
                if self.nodes[node][3][1][input_arc] is not None:
                    self.nodes[node][2] += self.recurrent_connections[self.nodes[node][3][1][input_arc]]
        for node in self.correct_node_order:
            try:
                self.nodes[node][2] = self.nodes[node][1](self.nodes[node][2])
            except OverflowError:
                self.nodes[node][2] = sys.float_info.max
            for connection in self.nodes[node][0]:
                if self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]] is None:
                    try:
                        self.nodes[self.unique_id_to_index[connection[2]]][2] += self.nodes[node][2] * connection[1]
                    except TypeError:
                        print("huh?")
                        return
                else:
                    self.recurrent_connections[self.nodes[self.unique_id_to_index[connection[2]]][3][1][connection[0]]] = self.nodes[node][2] * connection[1]
        tmp = self.precalculated_outputs
        for input_i in inputs[:-1]:
            if input_i in tmp:
                tmp = tmp[input_i]
            else:
                tmp[input_i] = {}
                tmp = tmp[input_i]
        outputs = list(map(lambda x: x[2], self.nodes[self.in_count: self.in_count + self.out_count]))
        outputs = list(map(lambda x: 0 if -0.2<x<0.2 else x, outputs))
        outputs = list(map(lambda x: -1 if x < -1 else x, outputs))
        outputs = list(map(lambda x: 1 if x > 1 else x, outputs))
        outputs = list(map(lambda x: x * 3, outputs))
        tmp[inputs[-1]] = outputs
        return tmp[inputs[-1]]

    def print_graph_data(self):
        for node in range(len(self.nodes)):
            print (self.index_to_unique(node))
        for node in range(len(self.nodes)):
            for connection in self.nodes[node][0]:
                print(str(node) + " " + str(connection[2]))

    def get_not_output(self):
        return list(range(self.in_count)) + list(range(self.in_count + self.out_count, len(self.nodes)))

    def mut_add_node(self, innov, node_unique_id):
        connections = []
        for node in range(len(self.nodes)):
            connections.extend([[node, connection] for connection in self.nodes[node][0]])
        chosen_arc = random.choice(connections)
        self.nodes.append(
            [[[innov+1, 1, chosen_arc[1][2]]], get_random_activation(), 0, [1, {innov: None}], 0,
             node_unique_id])
        self.unique_id_to_index[node_unique_id] = len(self.nodes) - 1
        self.nodes[self.unique_id_to_index[chosen_arc[1][2]]][3][1].pop(chosen_arc[1][0])
        self.nodes[self.unique_id_to_index[chosen_arc[1][2]]][3][1][innov+1] = None
        chosen_arc[1][2] = node_unique_id
        chosen_arc[1][0] = innov
        self.correct_node_order = None
        self.precalculated_outputs = {}

    def mut_add_connection(self, innov):
        # find list of possible sources
        source_list = []
        for node in self.get_not_output():
            if len(self.nodes[node][0]) < len(self.nodes) - self.in_count:
                source_list.append(node)
        if len(source_list) == 0:
            return
        source_node = random.choice(source_list)
        possible_connections = list(self.unique_id_to_index.keys())[self.in_count:]
        source_node = self.nodes[source_node]
        for connection in source_node[0]:
            try:
                possible_connections.remove(connection[2])
            except ValueError:
                continue
        if len(possible_connections) == 0:
            raise ValueError("This shouldn'tmai happen")
        new_target = random.choice(possible_connections)
        source_node[0].append([innov, random.gauss(0, 0.5), new_target])
        self.nodes[self.unique_id_to_index[new_target]][3][0] += 1
        self.nodes[self.unique_id_to_index[new_target]][3][1][innov] = None
        if not self.allow_recurrent_connections and self.is_node_dependent_on_self(new_target):
            self.nodes[self.unique_id_to_index[new_target]][3][0] += 1
            self.nodes[self.unique_id_to_index[new_target]][3][1].pop(innov)
            source_node[0].pop()
            return
        self.correct_node_order = None
        self.precalculated_outputs = {}

    def mut_weight(self):
        for node in self.get_not_output():
            for connection in self.nodes[node][0]:
                connection[1] += random.gauss(0, 0.1)
        self.precalculated_outputs = {}

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
        return self.nodes[self.unique_id_to_index[index]][1]

    def __contains__(self, item):
        return item in self.unique_id_to_index



class network_file_database:
    def __init__(self, filename, in_count, out_count):
        self.networks_as_strings = []
        line_no = 0
        with open(filename, "r") as networks_file:
            for line in networks_file:
                if line == "new individual\n":
                    self.networks_as_strings.append([])
                else:
                    self.networks_as_strings[-1].append(str(line_no) + "%" + line)
                line_no += 1
        self.networks = []
        for string_network in self.networks_as_strings:
            for string_node in range(len(string_network[:-1])):
                string_network[string_node]= string_network[string_node][:string_network[string_node].index("<")] + ("CPPN.ACTIVATION_FUNCTIONS['identity']" \
                    if "identity" in string_network[string_node] else "CPPN.ACTIVATION_FUNCTIONS['sigmoid']") + string_network[string_node][string_network[string_node].index(">") + 1:]
                #string_node.replace("<function identity_activation at 0x7f8cd7e663b0>", "CPPN.ACTIVATION_FUNCTIONS['identity']")
                #string_node.replace("<function sigmoid_activation at 0x7f8cd7e623b0>", "CPPN.ACTIVATION_FUNCTIONS['sigmoid']")
            network_as_list = []
            for string_node in string_network:
                #print(string_node)
                network_as_list.append(eval(string_node.split("%")[1]))
            self.networks.append(Network(network_as_list, in_count, out_count, None, len(network_as_list)**2))

if __name__ == "__main__":
    nfd = network_file_database("output_networks.ann", 25, 4)
    nfd.networks[-2].print_graph_data()
    testing.bipedal_walker(nfd.networks[-2], True, False, True)