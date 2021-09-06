import time
from copy import deepcopy as dc
from copy import copy as cc
import random
import numpy as np

from numba import cuda

def mutate_and_evaluate(individual, testing_function, mutation_function, testing_args, mutation_args, discard):
    individual = dc(individual)
    if mutation_args > 0:
        individual = mutation_function(individual)
    fitness = testing_function(individual, testing_args)
    return individual, fitness



def worker(queue_in, queue_out, new_networks):
    while True:
        value_received = queue_in.get()
        if value_received is None:
            break
        if len(value_received) == 3:
            queue_out.put([value_received[0], value_received[1](value_received[-1])])
        else:
            value_received[-1] = value_received[1](value_received[-1], *value_received[2])
            value_received = value_received[:1] + value_received[3:]
            queue_in.put(value_received)
            new_networks.put([value_received[0], value_received[-1]])


# receives a list in queue, with job_id, function, parameters
# returns job_id, result
def new_worker(queue_in, queue_out):
    while True:
        jobs_in = queue_in.get()
        if jobs_in is None:
            break
        todays_job = jobs_in[0]
        out = todays_job[1](*todays_job[2:])
        if todays_job[0] is not None:
            queue_out.put([todays_job[0], out])
        if len(jobs_in) > 1:
            tomorrows_job = jobs_in[1:]
            tomorrows_job[0].append(out)
            queue_in.put(jobs_in[1:])


def run_all_conv_networks_on_data(conv_networks, test_data):
    threads_per_block = 5
    blocks_per_grid = test_data.shape[0] * conv_networks.shape[0]
    meta_data = np.zeros((len(conv_networks), 10))
    max_height = test_data.shape[1]
    max_width = test_data.shape[2]



@cuda.jit
def run_convs_on_data_GPU(conv_networks, test_data, meta_data):
    return



