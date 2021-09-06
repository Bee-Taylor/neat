#import snakeProblem
import numpy
#from PIL import Image

#from ArtificialNetwork import Network
import latinCharEnvironment
import random
import time
#from multiprocessing import Pool
from multiprocessing import Process
from multiprocessing import Queue
#from copy import deepcopy as dc
#from workers import new_worker as worker
from workers import mutate_and_evaluate as mae
import birdEnvironment
import catvsdog
#import my_chess
#import math
#import testing
#import foraging_task
#import t_maze
##from substrate import Substrate
#import omniglot
#import snakeProblem
#import my_chess
#import conv_network
import conv_network_three as conv_network_new
from dask.distributed import Client
from dask.distributed import LocalCluster
import omniglot
import CIFAR_Environment as cifar

class Evolution_Manager:
    def __init__(self, process_count, population_size, survival_size, number_of_species, plasticity,
                 in_count, out_count, infinite, num_gens, testing_function, demonstration_function, branch_count,
                 tournament_size, diffusion_gradients):
        self.process_count = process_count
        self.population_size = population_size
        self.survival_size = survival_size
        self.number_of_species = number_of_species
        self.plasticity = plasticity
        self.in_count = in_count
        self.out_count = out_count
        self.infinite = infinite
        self.num_gens = num_gens
        self.testing_function = testing_function
        self.demonstration_function = demonstration_function
        self.branch_count = branch_count
        self.current_job_id = 0
        self.my_jobs = []
        self.tournament_size = tournament_size
        self.pop = []
        self.current_id = None
        self.number_deleted = 0
        self.innov_number = (in_count + 1) * out_count
        self.diffusion_gradients = diffusion_gradients

    def run(self, best_ind = None, filename = None, current_gen = 0):
        cluster = LocalCluster(n_workers=self.process_count, threads_per_worker=1, memory_limit=20e9)
        client = Client(cluster)
        raw = cifar.load_all()
        images_future = client.scatter(raw)
        #omni = client.submit(omniglot.make_new_omni, images_future, 5)
        if filename is None:
            filename = str(random.randint(0, 100))
        if best_ind is None:
            best_ind = [conv_network_new.conv_network(32, 32, 3, self.out_count) for i in range(self.population_size * 10)]
        pop = []
        big_future = client.scatter(best_ind, broadcast=True)
        previous_mut_count = -1
        gen_input = current_gen
        while current_gen < self.num_gens:
            pop = []
            gen_start_time = time.time()
            #omni = client.gather(omni)
            #omni_future = client.scatter(omni, broadcast=True)
            if False:
                as_range = self.population_size * 10
            else:
                as_range = self.population_size
            for a in range(as_range):
                if current_gen > 0 or type(best_ind) == conv_network_new:
                    pop.append(client.submit(mae, big_future, self.testing_function, conv_network_new.mutate_individual,
                                             images_future, a, a))
                else:
                    pop.append(client.submit(mae, best_ind[a], self.testing_function, conv_network_new.mutate_individual,
                                             images_future, a, a))
            #catvsdog.replaceOne(raw)
            #images_future = client.scatter(raw)
            #omni = client.submit(omniglot.make_new_omni, images_future, 5)
            if current_gen > gen_input:
                x = client.gather(x)
                y = client.gather(y)
            pop = client.gather(pop)
            del big_future
            fitnesses = list(map(lambda x: x[1], pop))
            out_string = str(current_gen) + ", " + str(max(fitnesses)) + ", " + str(numpy.mean(fitnesses)) + ", " + str(
                time.time() - gen_start_time) + ", pop size= " + str(len(fitnesses))
            #print(out_string)
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
            #with open(filename + "indepth", "w") as out_file:
            #    out_file.write(str(fitnesses) + "\n")
            #    #print("\t" + str(fitnesses))
            if not self.infinite:
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
    em = Evolution_Manager(process_count, population_size, survival_size, number_of_species, plasticity, in_count,
                           out_count, infinite, num_gens, testing_function, demonstration_function, branch_count,
                           tournament_size, diffusion_gradients)
    em.run()

