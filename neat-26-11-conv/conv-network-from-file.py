import conv_network_three
import ArtificialNetwork
import json
import birdEnvironment
import CIFAR_Environment as cifar
import latinCharEnvironment
from copy import deepcopy as dc
import numpy as np

import main

def make_network(network_as_dict):
    network_nodes = []
    if network_as_dict["rule_child"] is not None:
        network_as_dict["rule_child"] = make_network(network_as_dict["rule_child"])
    for node in network_as_dict["nodes"]:
        node[1] = eval(node[1])
        new_dict = {}
        for i in node[3][1]:
            new_dict[int(i)] = node[3][1][i]
        node[3][1] = new_dict
        network_nodes.append(node)
    network = ArtificialNetwork.Network(network_as_dict["original_in_count"], network_as_dict["out_count"],
                    network_as_dict["allow_recurrent_connections"], has_child=network_as_dict["rule_child"] is not None,
                    diffusion_gradients=network_as_dict["diffusion_gradients"], time_series=network_as_dict["time_series"])
    network.nodes = network_nodes
    network.in_count = network_as_dict["in_count"]
    network.rule_child = network_as_dict["rule_child"]
    network.input_ids = network_as_dict["in_ids"]
    return network


def ann_string_to_network(ann_string):
    #og_ann_string = dc(ann_string)
    #ann_string = ann_string.replace('width', '"width"')
    #ann_string = ann_string.replace('height', '"height"')
    #ann_string = ann_string.replace('depth', '"depth"')
    #ann_string = ann_string.replace('section_count', '"section_count"')
    #ann_string = ann_string.replace('big_network', '"big_network"')
    #ann_string = ann_string.replace('pane_count_in', '"pane_count_in"')
    #ann_string = ann_string.replace('pane_count_out', '"pane_count_out"')
    #ann_string = ann_string.replace('max_pool', '"max_pool"')
    #ann_string = ann_string.replace('filters', '"filters"')
    #ann_string = ann_string.replace('network:', '"network":')
    #ann_string = ann_string.replace('rule_child', '"rule_child"')
    #ann_string = ann_string.replace('allow_recurrent_connections', '"allow_recurrent_connections"')
    #ann_string = ann_string.replace('diffusion_gradients', '"diffusion_gradients"')
    #ann_string = ann_string.replace('time_series', '"time_series"')
    #ann_string = ann_string.replace('in_ids', '"in_ids"')
    #ann_string = ann_string.replace('frame_size', '"frame_size"')
    #ann_string = ann_string.replace('function_set.identity', '"ArtificialNetwork.ACTIVATION_FUNCTIONS[\'identity\']"')
    #ann_string = ann_string.replace('function_set.relu', '"ArtificialNetwork.ACTIVATION_FUNCTIONS[\'relu\']"')
    #ann_string = ann_string.replace('function_set.sigmoid', '"ArtificialNetwork.ACTIVATION_FUNCTIONS[\'sigmoid\']"')
    #ann_string = ann_string.replace('False', 'false')
    #ann_string = ann_string.replace('nodes', '"nodes"')
    #ann_string = ann_string.replace(' in_count', ' "in_count"')
    #ann_string = ann_string.replace('original_in_count', '"original_in_count"')
    #ann_string = ann_string.replace('out_count', '"out_count"')
    #ann_string = ann_string.replace('None', 'null')
    #ann_string = ann_string.replace('True', 'true')
    #ann_string = ann_string.replace('final_inputs_per_filter', '"final_inputs_per_filter"')
    #ann_string = ann_string.replace('final_layer', '"final_layer"')
    #ann_string = ann_string.replace(', ]', ']')
    #for i in range(10000):
    #    ann_string = ann_string.replace(" " + str(i) + ":", ' "' + str(i) + '":')
    #    ann_string = ann_string.replace("{" + str(i) + ":", '{"' + str(i) + '":')
    decoder = json.JSONDecoder()
    ann_json = decoder.decode(ann_string)

    big_network = []
    for section in ann_json["big_network"]:
        if section["max_pool"]:
            big_network.append(conv_network_three.Max_Pool_Section(False))
        else:
            big_network.append(conv_network_three.Inception(0,0,0,0))
            big_network[-1].C1F1 = np.array(section["C1F1"], dtype=float)
            big_network[-1].C3F1 = np.array(section["C3F1"], dtype=float)
            big_network[-1].C5F1 = np.array(section["C5F1"], dtype=float)
            big_network[-1].C7F1 = np.array(section["C7F1"], dtype=float)
            big_network[-1].C3F2 = np.array(section["C3F2"], dtype=float)
            big_network[-1].C5F2 = np.array(section["C5F2"], dtype=float)
            big_network[-1].C7F2 = np.array(section["C7F2"], dtype=float)
            big_network[-1].CMF2 = np.array(section["CMF2"], dtype=float)

    final_layer = conv_network_three.Inception(0,0,0,0)
    final_layer.C1F1 = np.array(ann_json["final_layer"]["C1F1"], dtype=float)
    final_layer.C3F1 = np.array(ann_json["final_layer"]["C3F1"], dtype=float)
    final_layer.C5F1 = np.array(ann_json["final_layer"]["C5F1"], dtype=float)
    final_layer.C7F1 = np.array(ann_json["final_layer"]["C7F1"], dtype=float)
    final_layer.C3F2 = np.array(ann_json["final_layer"]["C3F2"], dtype=float)
    final_layer.C5F2 = np.array(ann_json["final_layer"]["C5F2"], dtype=float)
    final_layer.C7F2 = np.array(ann_json["final_layer"]["C7F2"], dtype=float)
    final_layer.CMF2 = np.array(ann_json["final_layer"]["CMF2"], dtype=float)

    conv_network = conv_network_three.conv_network(ann_json["width"], ann_json["height"], ann_json["depth"], ann_json["out_count"])
    conv_network.big_network = big_network
    conv_network.final_layer = final_layer
    return conv_network


def open_network():
    file = open("dogcatdata/47bestindgen8613.json")
    ann_string = ""
    for line in file:
        ann_string += line
    return ann_string


#conv_network = conv_network_new.conv_network(224, 224, 3, 4, 4)
#for i in range(10):
#    conv_network = conv_network_new.mutate_individual(conv_network)
#conv_network_i = ann_string_to_network(repr(conv_network))

if __name__ == "__main__":
    conv_network = ann_string_to_network(open_network())

    import CIFAR_Environment
    tmp = CIFAR_Environment.load_all()

    from numba import cuda
    tmp = cuda.to_device(tmp)

    #print(birdEnvironment.evaluate_agent_non_learning(conv_network, tmp))
    #print(birdEnvironment.evaluate_agent_non_learning(conv_network, birdEnvironment.open_shapes()))
    #print(birdEnvironment.evaluate_agent_on_all(conv_network, tmp))


    process_count = 10
    population_size = 10
    survival_size = 2
    branch_count = 50
    number_of_species = 5
    plasticity = True
    in_count = 65
    out_count = 10
    infinite = False
    num_gens = 100000  # if infinite, this variable doesn't matter (but keep it > 0)
    tournament_size = 3
    diffusion_gradients = 50
    testing_function = birdEnvironment.evaluate_agent_non_learning
    demonstration_function = birdEnvironment.demonstrate
    print("entering main")
    em = main.Evolution_Manager(process_count, population_size, survival_size, number_of_species, plasticity, in_count,
                           out_count, infinite, num_gens, testing_function, demonstration_function, branch_count,
                           tournament_size, diffusion_gradients)
    #em.run(conv_network, "47", 2645)
    raw = cifar.load_all()
    print(testing_function(conv_network, raw))
