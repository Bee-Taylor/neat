import random
import tkinter as tk
import time
from copy import deepcopy as dc
import numpy as np

roots = [None]
class foraging_task:
    def __init__(self, number_of_bits, manual_input = None):
        if number_of_bits < 2:
            raise ValueError("There must be at least 2 bits in an encoding")
        self.number_of_bits = number_of_bits
        tmp = list(range(self.number_of_bits))
        if manual_input is None:
            self.summer_poison = random.choice(tmp)
            self.summer_orientation = random.getrandbits(1)
            self.winter_poison = random.choice(tmp)
            self.winter_orientation = random.getrandbits(1)
        else:
            self.summer_poison = manual_input[0]
            self.summer_orientation = manual_input[2]
            self.winter_poison = manual_input[1]
            self.winter_orientation = manual_input[3]

        self.premade_inputs = [[random.getrandbits(1) for i in range(self.number_of_bits)] for j in range(30)]

    def run(self, agent, render):
        if render:
            global roots
            if roots[0] is None:
                roots[0] = tk.Tk()
            root = roots[0]
            root.geometry("430x715")
            texts =[[tk.Text(root, height = 1, width = 10) for i in range(5)] for j in range(31)]
            for row in range(31):
                for column in range(5):
                    texts[row][column].grid(row = row, column = column)
            texts[0][0].insert(tk.END, "Season")
            texts[0][1].insert(tk.END, "Input")
            texts[0][2].insert(tk.END, "Choice")
            texts[0][3].insert(tk.END, "Reward")
            texts[0][4].insert(tk.END, "Fitness")
        cum_reward = 0
        counter = 0
        for year in range(3):
            for season in [True, False]:
                for day in range(5):
                    input = self.premade_inputs[day + (0 if season else 5) + year * 10]
                    output = agent.run(list(map(lambda x: -1 if x == 0 else x, input)))
                    if render:
                        root.update()
                        time.sleep(1)
                        if season:
                            texts[counter+1][0].insert(tk.END, "Summer")
                        else:
                            texts[counter+1][0].insert(tk.END, "Winter")
                        texts[counter + 1][1].insert(tk.END, str(input)[1:-1])
                        if season and output[0] > output[1] or not season and output[2] > output[3]:
                            texts[counter+1][2].insert(tk.END, "Leave")
                            texts[counter+1][3].insert(tk.END, "0")
                            texts[counter+1][4].insert(tk.END, str(cum_reward))
                        else:
                            texts[counter+1][2].insert(tk.END, "Eat")
                    counter += 1
                    if season and output[0] > output[1] or not season and output[2] > output[3]:
                        continue
                    else:
                        if season:
                            if input[self.summer_poison] == self.summer_orientation:
                                cum_reward += 1
                                agent.learn([1,0])
                                if render:
                                    texts[counter][3].insert(tk.END, "+1")
                            else:
                                cum_reward -= 1
                                agent.learn([-1,0])
                                if render:
                                    texts[counter][3].insert(tk.END, "-1")
                        else:
                            if input[self.winter_poison] == self.winter_orientation:
                                cum_reward += 1
                                agent.learn([0,1])
                                if render:
                                    texts[counter][3].insert(tk.END, "+1")
                            else:
                                cum_reward -= 1
                                agent.learn([0,-1])
                                if render:
                                    texts[counter][3].insert(tk.END, "-1")
                    if render:
                        texts[counter][4].insert(tk.END, cum_reward)
                        root.update()
        if render:
            root.update()
            time.sleep(1)
        return cum_reward

    def human_player(self):
        cum_reward = 0
        for year in range(3):
            #print(year)
            for season in [True, False]:
                print(season)
                for day in range(5):
                    stuff = [random.getrandbits(1) for i in range(self.number_of_bits)]
                    print(stuff)
                    ui = input()
                    if ui == "1":
                        if season:
                            if stuff[self.summer_poison] == self.summer_orientation:
                                cum_reward += 1
                                print("+")
                            else:
                                print("-")
                        else:
                            if stuff[self.winter_poison] == self.winter_orientation:
                                cum_reward += 1
                                print("+")
                            else:
                                cum_reward -= 1
                                print("-")
        print(cum_reward)

    def reset(self):
        tmp = list(range(self.number_of_bits))
        self.summer_poison = random.choice(tmp)
        self.summer_orientation = random.getrandbits(1)
        self.winter_poison = random.choice(tmp)
        self.winter_orientation = random.getrandbits(1)
        self.premade_inputs = [[random.getrandbits(1) for i in range(self.number_of_bits)] for j in range(30)]

ft = foraging_task(4)

def new_gen():
    ft.reset()

def evaluate_agent(agent):
    fitness = 0
    number_of_bits = 4
    combinations = np.array(
        np.meshgrid(
            [i for i in range(number_of_bits)],
            [i for i in range(number_of_bits)],
            [0, 1], [0, 1])
    ).T.reshape(-1, 4)
    combinations = random.sample(list(combinations), 4)
    for i in combinations:
        ft_i = foraging_task(number_of_bits, i)
        fitness += ft_i.run(dc(agent), False)
    return fitness

def show_agent(agent):
    ft_i = foraging_task(4)
    ft_i.run(dc(agent), True)


def player(queue):
    keep_going = True
    first = True
    while keep_going:
        if not queue.empty() or first:
            keep_going, network, choice = queue.get()
        first =  False
        hard = bool(choice%2)
        distance = bool(choice/2%1)
        show_agent(dc(network))