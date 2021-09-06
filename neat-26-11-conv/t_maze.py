import random
from shapely.geometry import Polygon, Point
from copy import copy as cc
from copy import deepcopy as dc
#import tkinter as tk
import time
from functools import reduce

def inside_rectangle(point, bottom_left, top_right):
    return bottom_left[0] < point[0] < top_right[0] and bottom_left[1] < point[1] < top_right[1]


"""
The t maze is a test bed usually used for meta-learning and testing plasticity
This t maze has 4 parameters:
- levels: an integer, how many levels of t maze are there (single, double, triple etc.)
- wall_death: boolean, does the agent die when it collides with a wall?
for now, this is only a single t-maze
"""
roots = [None]
class T_Maze:
    def __init__(self, levels, wall_death, position = None):
        self.levels = levels
        self.wall_death = wall_death
        self.agent_coordinates = [250, 475]
        self.last_coordinates = [250,475]
        if position is None:
            self.correct_choice = random.randint(0, 2**levels - 1)
        else:
            self.correct_choice = position
        self.run_length = 150
        if levels == 1:
            self.walls = [[200,495],[200,300],[5,300],[5,200],[495,200],[495,300],[300,300],[300,495]]
            self.walls_flat = [200,495,200,300,5,300,5,200,495,200,495,300,300,300,300,495]
            self.walls_polygon = Polygon(self.walls)
            self.finished_zones = [[[5,200],[50,200],[50,300],[5,300]], [[450,200],[495,200],[495,300],[450,300]]]
            self.finished_polygons = [Polygon(i) for i in self.finished_zones]
            self.finished_zones_flat = [[10,205,50,295], [450,205,490,295]]
        elif levels == 2:
            self.walls = [[200,495],[200,300],[100,300],[100,400],[5,400],[5,100],[100,100],[100,200],[400,200],[400,100],[495,100],[495,400],[400,400],[400,300],[300,300],[300,495]]
            self.walls_flat = [200,495,200,300,100,300,100,400,5,400,5,100,100,100,100,200,400,200,400,100,495,100,495,400,400,400,400,300,300,300,300,495]
            self.walls_polygon = Polygon(self.walls)
            self.finished_zones = [[[5,350], [5,400], [100,400], [100,350]],
                                   [[5,150], [5,100], [100,100], [100,150]],
                                   [[495, 350], [495, 400], [400, 400], [400, 350]],
                                   [[495, 150], [495, 100], [400, 100], [400, 150]]]
            self.finished_polygons = [Polygon(i) for i in self.finished_zones]
            self.finished_zones_flat = [[10, 350, 95, 395],
                                   [10, 150, 95, 105],
                                   [490, 350, 405, 395],
                                   [490, 150, 405, 105]]
        elif levels == 3:
            self.walls = [[210, 495], [210, 300], [125, 300], [125, 400], [175, 400], [175, 475], [100, 475],
                          [5, 475], [5, 400], [50, 400], [50, 100], [5, 100], [5, 25], [175, 25],
                          [175, 100], [125, 100], [125, 200], [375, 200], [375, 100], [350, 100], [325, 100], [325, 25], [495, 25],
                          [495, 100], [450, 100], [450, 400], [495, 400], [495, 475], [325, 475], [325, 400],
                          [375, 400], [375, 300],  [290, 300], [290, 495]]
            self.walls_flat = list(reduce(lambda x, y: x+y, self.walls))
            self.walls_polygon = Polygon(self.walls)
            self.finished_zones = [[[150, 405], [150, 470], [175, 470], [175, 405]],
                                   [[5, 405], [5, 470], [30, 470], [30, 405]],
                                   [[150, 25], [150, 100], [175, 100], [175, 25]],
                                   [[5, 25], [5, 100], [30, 100], [30, 25]],

                                   [[470, 405], [470, 470], [495, 470], [495, 405]],
                                   [[325, 405], [325, 470], [350, 470], [350, 405]],
                                   [[470, 25], [470, 100], [495, 100], [495, 25]],
                                   [[325, 25], [325, 100], [350, 100], [350, 25]]]
            self.finished_polygons = [Polygon(i) for i in self.finished_zones]
            self.finished_zones_flat = [[150, 405, 170, 470], [10, 405, 30, 470],
                                        [150, 30, 170, 95], [10, 30, 30, 95],
                                        [470, 405, 490, 470], [330, 405, 350, 470],
                                        [470, 30, 490, 95], [330, 30, 350, 95],

                                        ]
        else:
            raise ValueError("only 1 level is support rn")

    def reset_maze_soft(self):
        self.agent_coordinates = [250, 490]
        self.last_coordinates = cc(self.agent_coordinates)

    def reset_maze_hard(self):
        self.agent_coordinates = [250, 490]
        self.correct_choice = [bool(random.getrandbits(1)) for i in range(self.levels)]

    def agent_in_maze(self):
        return self.walls_polygon.contains(Point(self.agent_coordinates))

    def in_finished_zones(self):
        agent_point = Point(self.agent_coordinates)
        for zone in range(len(self.finished_zones)):
            if self.finished_polygons[zone].contains(agent_point):
            #if inside_rectangle(self.agent_coordinates, self.finished_zones[zone][0], self.finished_zones[zone][2]):
                if self.correct_choice == zone:
                    return True, 1, zone
                else:
                    return True, -1, zone
        return False, 0, None

    def get_inputs(self):
        inputs = [1 - self.agent_coordinates[0]/250, 1 - self.agent_coordinates[1]/250]
       # inputs.append(1-((self.agent_coordinates[0] - 200) / 250) if self.agent_coordinates[1] > 300 else
          #                  1 - self.agent_coordinates[0]/250)
        #inputs.append(1 - ((300 - self.agent_coordinates[0]) / 250) if self.agent_coordinates[1] > 300 else
         #             self.agent_coordinates[0] / 250 - 1)
        return inputs

    def one_run(self, agent, render, progress=0, human=False):
        self.reset_maze_soft()
        if render:
            global roots
            #root = tk.Tk()
            if roots[0] is None:
                roots[0] = tk.Tk()
            root = roots[0]
            root.geometry("500x500")
            canvas = tk.Canvas(root, width=500, height=500)
            canvas.configure(background="#aaa")
            canvas.pack()
            canvas.create_polygon(self.walls_flat, fill="#eee", outline="black", width=10)
            player = canvas.create_rectangle(self.agent_coordinates[0]-10, self.agent_coordinates[1]-10,
                                             self.agent_coordinates[0]+10, self.agent_coordinates[1] + 10,
                                             fill="black")
            for zone in range(len(self.finished_zones)):
                fill = "red"
                if zone == self.correct_choice:
                    fill = "green"
                canvas.create_rectangle(self.finished_zones_flat[zone], fill=fill, width=0)
            canvas.create_rectangle(0,5,(progress*self.run_length)/15,50, fill = "#666", width = 0)
            root.update()
        finished = False
        counter = 0
        explored = {}
        wall_counter = 0
        while not finished and counter < self.run_length:
            if not human:
                output = agent.run(self.get_inputs())
                for output_i in range(len(output)):
                    output[output_i] += output_i / 100000
                if wall_counter >= len(output):
                    wall_counter = len(output) - 1
                #choice = output.index(max(output))
                output_sorted = cc(output)
                output_sorted.sort()
                choice = output.index(output_sorted[wall_counter])
            else:
                print("choose")
                choice = int(input())
            self.last_coordinates = cc(self.agent_coordinates)
            if self.last_coordinates[0] not in explored:
                explored[self.last_coordinates[0]] = [self.last_coordinates[1]]
            elif self.last_coordinates[1] not in explored[self.last_coordinates[0]]:
                explored[self.last_coordinates[0]].append(self.last_coordinates[1])
            if choice == 0:
                self.agent_coordinates[0] -= 10
            elif choice == 1:
                self.agent_coordinates[0] += 10
            elif choice == 2:
                self.agent_coordinates[1] -= 10
            elif choice == 3:
                self.agent_coordinates[1] += 10
            if not self.agent_in_maze():
                self.agent_coordinates = self.last_coordinates
                wall_counter += 1
                if self.wall_death:
                    finished = True
            else:
                wall_counter = 0
            if render:
                canvas.delete(player)
                player = canvas.create_rectangle(self.agent_coordinates[0]-10,self.agent_coordinates[1]-10,
                                             self.agent_coordinates[0]+10, self.agent_coordinates[1] + 10,
                                                 fill = "black")
                root.update()
                canvas.create_rectangle(0,5,(counter + progress*self.run_length)/15,50, fill = "#666", width = 0)
                time.sleep(0.01)
            zone_interraction, reward, zone = self.in_finished_zones()
            if zone_interraction:
                finished = True
            counter += 1
        if render:
            canvas.destroy()
        if reward > 0:
            reward = reward * (progress + 1)
        to_return = [reward] + [0 for i in range(2**self.levels)]
        if zone_interraction:
            to_return[zone+1] = reward
        else:
            if progress == 0:
                explored_length = 0
                for i in explored.keys():
                    explored_length += len(explored[i])
                to_return[0] = -100000/explored_length
            else:
                explored_length = 0
                for i in explored.keys():
                    explored_length += len(explored[i])
                to_return[0] = -1/explored_length
        return to_return


def evaluate_agent(agent):
    depth = 2
    agent_clones = [dc(agent) for i in range(depth)]
    t_mazes = [T_Maze(1, False, random.randint(0,7)) for i in range(depth)]
    cum_reward = 0
    for id in range(depth):
        for i in range(20):
            reward_tmp = t_mazes[id].one_run(agent_clones[id], False, progress = i)
            agent_clones[id].learn(reward_tmp[1:])
            cum_reward += reward_tmp[0]
    return cum_reward, 0

def show_agent(agent):
    t_maze = T_Maze(1, False)
    agent_clone = dc(agent)
    cum_reward = 0
    for i in range(20):
        reward_tmp = t_maze.one_run(agent_clone, True, progress= i)
        agent_clone.learn(reward_tmp[1:])
        cum_reward += reward_tmp[0]
    return cum_reward

def player(queue):
    keep_going = True
    first = True
    while keep_going:
        if not queue.empty() or first:
            keep_going, network, choice = queue.get()
        first = False
        hard = bool(choice%2)
        distance = bool(choice/2%1)
        show_agent(network)

global_val = 0
class testing_agent:
    def __init__(self):
        self.counter = 0
        self.x = 0
        self.uppy = random.choice([[1,0],[0,1]])
        self.downy = random.choice([[1,0],[0,1]])

    def run(self, inputs):
        self.counter += 1
        if self.counter < 21:
            return [random.gauss(0,0.1), random.gauss(0,0.1), 1, random.gauss(0,0.1)]
        elif self.counter < 40:
            return self.uppy + [random.gauss(0,0.1), random.gauss(0,0.1)]
        else:
            return [random.gauss(0,0.1), random.gauss(0,0.1)] + self.downy

    def learn(self, reward):
        if max(reward) <= 0:
            self.uppy = random.choice([[1,0],[0,1]])
            self.downy = random.choice([[1,0],[0,1]])
        self.counter = 0
        print(reward)
        return None


if __name__ == "__main__":
    print("huh?")
    t_maze = T_Maze(3, True)
    agent = testing_agent()
    sum = 0
    count = 0
    if True:
        #print(t_maze.one_run(agent, True, human=False, progress=agent.x))
        print(t_maze.one_run(agent, human=True, render=True))
        agent.counter = 0
        agent.x += 1
        #print(str(count) + ": " + str(sum))
        count += 1
