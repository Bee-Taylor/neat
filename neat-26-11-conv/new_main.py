import numpy
import random
import time
import birdEnvironment
import conv_network_three as conv_network_new
from dask.distributed import Client
from dask.distributed import LocalCluster
import CIFAR_Environment as cifar
from copy import deepcopy as dc


def mutate_and_evaluate(individual, testing_function, mutation_function, testing_args, mutation_args, target):
    individual = dc(individual)
    if mutation_args > 0:
        individual = mutation_function(individual)
    fitness = testing_function(individual, testing_args, target)
    return individual, fitness

class Evolution_Manager:
    def __init__(self, process_count, population_size, plasticity, out_count, testing_function, demonstration_function, diffusion_gradients):
        self.process_count = process_count
        self.population_size = population_size
        self.plasticity = plasticity
        self.out_count = out_count
        self.testing_function = testing_function
        self.demonstration_function = demonstration_function
        self.current_job_id = 0
        self.my_jobs = []
        self.pop = []
        self.current_id = None
        self.number_deleted = 0
        self.diffusion_gradients = diffusion_gradients

    def run(self, best_ind = None, filename = None, classification_target = None, current_gen = 0):
        cluster = LocalCluster(n_workers=self.process_count, threads_per_worker=1, memory_limit=20e9)
        client = Client(cluster)
        raw = cifar.load_all()
        images_future = client.scatter(raw)
        if filename is None:
            filename = str(random.randint(0, 100))
        if best_ind is None:
            best_ind = conv_network_new.conv_network(32, 32, 3, 2)
        big_future = client.scatter(best_ind, broadcast=True)
        previous_mut_count = -1
        gen_input = current_gen
        while True:
            pop = []
            gen_start_time = time.time()
            for a in range(self.population_size):
                pop.append(client.submit(mutate_and_evaluate, big_future, self.testing_function, conv_network_new.mutate_individual,
                                         images_future, a, a))
            if current_gen > gen_input:
                x = client.gather(x)
                y = client.gather(y)
            pop = client.gather(pop)
            del big_future
            fitnesses = list(map(lambda x: x[1], pop))

            out_string = str(current_gen) + ", " + str(max(fitnesses)) + ", " + str(numpy.mean(fitnesses)) + ", " + str(
                time.time() - gen_start_time) + ", pop size= " + str(len(fitnesses))

            best_ind = pop[fitnesses.index(max(fitnesses))][0]
            if best_ind.mutation_count == previous_mut_count:
                best_ind.mutation_sd *= 0.9
                best_ind.decay()
            else:
                best_ind.mutation_sd = 0.1
                previous_mut_count = best_ind.mutation_count
            print("best ind mutation count: " + str(best_ind.mutation_count))
            big_future = client.scatter(best_ind, broadcast=True)
            x = client.submit(writeBestToFile, big_future, filename, current_gen)
            y = client.submit(writeToFile, big_future, out_string, images_future, filename)
            current_gen += 1


def writeToFile(best, out_string, examples, filename):
    percentage = birdEnvironment.evaluate_agent_non_learning(best, examples, percentage=True)
    out_string += ", " + str(percentage) + "\n"
    print(out_string)
    with open("dogcatdata/" + filename, "a") as out_file:
        out_file.write(out_string + "\n")


def writeBestToFile(best, filename, gen):
    with open("dogcatdata/" + filename + "bestindgen" + str(gen) + ".json", "w") as out_file:
        out_file.write(repr(best))

if __name__ == "__main__":
    process_count = 6
    population_size = 5
    plasticity = True
    out_count = 10
    diffusion_gradients = 0
    testing_function = birdEnvironment.evaluate_agent_non_learning
    demonstration_function = birdEnvironment.demonstrate
    em = Evolution_Manager(process_count, population_size, plasticity,
                           out_count, testing_function, demonstration_function, diffusion_gradients)
    cluster = LocalCluster(n_workers=out_count, threads_per_worker=1, memory_limit=100e9)
    em.run()

