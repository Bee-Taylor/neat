# This code defines the agent (as in the playable version) in a way that can be called and executed from an evolutionary algorithm. The code is partial and will not execute. You need to add to the code to create an evolutionary algorithm that evolves and executes a snake agent.
#import curses
#from curses import KEY_LEFT
import random
import operator
from functools import partial
import time
from multiprocessing import Process

import numpy

#from deap import algorithms
#from deap import base
#from deap import creator
#from deap import tools
#from deap import gp

from copy import deepcopy as dc

import math

import tkinter as tk

S_RIGHT, S_LEFT, S_UP, S_DOWN = 0, 1, 2, 3
NFOOD = 1  # NOTE: YOU MAY NEED TO ADD A CHECK THAT THERE ARE ENOUGH SPACES LEFT FOR THE FOOD (IF THE TAIL IS VERY LONG)


# This class can be used to create a basic player object (snake agent)
# A few of the new sensing functions I added were not used but intentionally left in for future work
class SnakePlayer(list):
    global S_RIGHT, S_LEFT, S_UP, S_DOWN

    def __init__(self, XSIZE, YSIZE):
        self.XSIZE, self.YSIZE = XSIZE, YSIZE
        self.direction = S_RIGHT
        # self.body = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.body = [[random.randint(0, self.XSIZE - 1), random.randint(0, self.YSIZE - 1)]]
        self.score = 0
        self.ahead = []
        self.food = []

    def _reset(self):
        self.direction = S_RIGHT
        # self.body[:] = [[4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3], [4, 2], [4, 1], [4, 0]]
        self.body = [[random.randint(0, self.XSIZE - 1), random.randint(0, self.YSIZE - 1)]]
        self.score = 0
        self.ahead = []
        self.food = []

    def changeDirectionRightRel(self):
        if self.direction == S_UP:
            self.direction = S_RIGHT
        elif self.direction == S_RIGHT:
            self.direction = S_DOWN
        elif self.direction == S_DOWN:
            self.direction = S_LEFT
        elif self.direction == S_LEFT:
            self.direction = S_UP

    def changeDirectionLeftRel(self):
        if self.direction == S_UP:
            self.direction = S_LEFT
        elif self.direction == S_RIGHT:
            self.direction = S_UP
        elif self.direction == S_DOWN:
            self.direction = S_RIGHT
        elif self.direction == S_LEFT:
            self.direction = S_DOWN

    # ONLY CALL WHEN DOING NOVELTY SEARCH
    def make_random_snake_board(self):
        self.body = []
        snake = [[random.choice(range(self.YSIZE)), random.choice(range(self.XSIZE))]]
        snake_length = random.randint(5, 194)
        for i in range(snake_length):
            coord = [random.choice(range(self.YSIZE)), random.choice(range(self.XSIZE))]
            while coord in snake:
                coord = [random.choice(range(self.YSIZE)), random.choice(range(self.XSIZE))]
            self.body.append(coord)
        coord = [random.choice(range(self.YSIZE)), random.choice(range(self.XSIZE))]
        while coord in self.body:
            coord = [random.choice(range(self.YSIZE)), random.choice(range(self.XSIZE))]
        self.food = [coord]

    def setAheadToHead(self):
        self.ahead = self.body[0]

    def move_ahead(self, vector):
        self.ahead = [self.ahead[0] + vector[0], self.ahead[1] + vector[1]]

    def getUpLocation(self):
        self.ahead = [self.body[0][0] - 1, self.body[0][1]]

    def getRightLocation(self):
        self.ahead = [self.body[0][0], self.body[0][1] + 1]

    def getLeftLocation(self):
        self.ahead = [self.body[0][0], self.body[0][1] - 1]

    def getDownLocation(self):
        self.ahead = [self.body[0][0] + 1, self.body[0][1]]

    def getAheadLocation(self):
        self.ahead = [self.body[0][0] + (self.direction == S_DOWN and 1) + (self.direction == S_UP and -1),
                      self.body[0][1] + (self.direction == S_LEFT and -1) + (self.direction == S_RIGHT and 1)]

    def updatePosition(self):
        self.getAheadLocation()
        self.body.insert(0, self.ahead)

    ## You are free to define more sensing options to the snake

    def changeDirectionUp(self):
        self.direction = S_UP

    def changeDirectionRight(self):
        self.direction = S_RIGHT

    def changeDirectionDown(self):
        self.direction = S_DOWN

    def changeDirectionLeft(self):
        self.direction = S_LEFT

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (self.YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
                self.XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return (self.hit)

    def sense_wall_at_ahead_loc(self):
        return self.ahead[0] == 0 or self.ahead[0] == (self.YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (
                    self.XSIZE - 1)

    def sense_tail_at_ahead_loc(self):
        return self.ahead in self.body

    def get_distance_to_obsticals(self):
        output = []
        vector = [-1, 1]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [0, 1]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [1, 1]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [1, 0]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [1, -1]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [0, -1]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [-1, -1]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        vector = [-1, 0]
        self.setAheadToHead()
        self.move_ahead(vector)
        output.append(float(1))
        while not (self.sense_wall_at_ahead_loc() or self.sense_tail_at_ahead_loc()):
            output[-1] += 1
            self.move_ahead(vector)
        for out in range(len(output)):
            output[out] = 1 - (output[out] / 14)
        return output

    def sense_walls(self):
        self.getUpLocation()
        to_return = [self.sense_wall_at_ahead_loc()]
        self.getDownLocation()
        to_return.append(self.sense_wall_at_ahead_loc())
        self.getLeftLocation()
        to_return.append(self.sense_wall_at_ahead_loc())
        self.getRightLocation()
        to_return.append(self.sense_wall_at_ahead_loc())
        return to_return

    def sense_tails(self):
        self.getUpLocation()
        to_return = [self.ahead in self.body]
        self.getDownLocation()
        to_return.append(self.ahead in self.body)
        self.getLeftLocation()
        to_return.append(self.ahead in self.body)
        self.getRightLocation()
        to_return.append(self.ahead in self.body)
        return to_return

    def sense_obsticals(self):
        walls = self.sense_walls()
        tails = self.sense_tails()
        to_return = []
        for i in range(len(walls)):
            to_return.append(walls[i] or tails[i])
        return to_return

    def get_board(self):
        board = []
        for x in range(self.XSIZE):
            board.append([])
            for y in range(self.YSIZE):
                if [x, y] in self.body:
                    board[x].append(-1)
                elif [x, y] == self.get_food_loc():
                    board[x].append(1)
                else:
                    board[x].append(0)
        return board

    def get_board_rel(self, win_size):
        board = self.get_board()
        head_loc = self.get_head_loc()
        window = []
        for row in range(head_loc[0] - win_size, head_loc[0] + win_size + 1):
            for column in range(head_loc[1] - win_size, head_loc[1] + win_size + 1):
                window.append([row, column])
        new_window = []
        for coordinate in window:
            if coordinate[0] < 0 or coordinate[1] < 0 or coordinate[0] >= self.XSIZE or coordinate[1] >= self.YSIZE:
                new_window.append(-1)
            else:
                new_window.append(board[coordinate[0]][coordinate[1]])
        return new_window

    def get_board_rel_2d(self, win_size):
        board = self.get_board()
        head_loc = self.get_head_loc()
        window = []
        for row in range(head_loc[0] - win_size, head_loc[0] + win_size + 1):
            window.append([])
            for column in range(head_loc[1] - win_size, head_loc[1] + win_size + 1):
                if column < 0 or row < 0 or column >= self.XSIZE or row >= self.YSIZE:
                    window[-1].append([0,0,1])
                else:
                    window[-1].append([board[row][column] == 1, board[row][column] == 0, board[row][column] == -1])
        return window

    def get_food_loc(self):
        return self.food

    def get_head_loc(self):
        return self.body[0]

    def find_angle_to_food_rel(self):
        food_vector_tmp = list(map(lambda x, y: float(x - y), self.get_food_loc()[0], self.get_head_loc()))
        food_vector_size = math.sqrt(food_vector_tmp[0] ** 2 + food_vector_tmp[1] ** 2)
        food_vector_tmp = [food_vector_tmp[0] / food_vector_size, food_vector_tmp[1] / food_vector_size]
        food_vector = [0, 0]
        if self.direction == S_LEFT:
            food_vector[0] = food_vector_tmp[1]
            food_vector[1] = -food_vector_tmp[0]
        elif self.direction == S_RIGHT:
            food_vector[0] = -food_vector_tmp[1]
            food_vector[1] = food_vector_tmp[0]
        elif self.direction == S_DOWN:
            food_vector[0] = -food_vector_tmp[1]
            food_vector[1] = -food_vector_tmp[0]
        else:
            food_vector = food_vector_tmp
        angle = 0
        if food_vector[1] > 0:
            angle = 0.5 + (food_vector[0] / 2)
        elif food_vector[1] < 0:
            angle = -0.5 - (food_vector[0] / 2)
        elif food_vector[1] == 0:
            if food_vector[0] < 0:
                angle = 0
            else:
                angle = 1
        return angle

    def find_angle_to_food(self):
        food_vector_tmp = list(map(lambda x, y: float(x - y), self.food[0], self.body[0]))
        food_vector_size = math.sqrt(food_vector_tmp[0] ** 2 + food_vector_tmp[1] ** 2)
        food_vector = [food_vector_tmp[0] / food_vector_size, food_vector_tmp[1] / food_vector_size]
        angle = 0
        if food_vector[1] > 0:
            angle = 0.5 + (food_vector[0] / 2)
        elif food_vector[1] < 0:
            angle = -0.5 - (food_vector[0] / 2)
        elif food_vector[1] == 0:
            if food_vector[0] < 0:
                angle = 0
            else:
                angle = 1
        return food_vector

    def get_direction(self):
        if self.direction == S_RIGHT:
            return [1, 0, 0, 0]
        elif self.direction == S_UP:
            return [0, 1, 0, 0]
        elif self.direction == S_LEFT:
            return [0, 0, 1, 0]
        elif self.direction == S_DOWN:
            return [0, 0, 0, 1]

    def snakeHasCollided(self):
        self.hit = False
        if self.body[0][0] == 0 or self.body[0][0] == (self.YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
                self.XSIZE - 1): self.hit = True
        if self.body[0] in self.body[1:]: self.hit = True
        return (self.hit)

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return (
                self.ahead[0] == 0 or self.ahead[0] == (self.YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (
                    self.XSIZE - 1))


def get_output_old(individual, snake):
    # print alternative_input
    # inputs = snake.sense_obsticals()
    # inputs  = snake.get_head_loc()
    # inputs.append(snake.direction)
    # inputs.extend(snake.get_food_loc())
    # inputs.extend(snake.get_head_loc())
    # alternative_input = [snake.get_head_loc()[0], snake.get_head_loc()[1], (snake.direction-2) / 4]#, snake.get_food_loc()[0], snake.get_food_loc()[1]]
    inputs = snake.get_board_rel()
    inputs.append(snake.find_angle_to_food())
    for input in range(len(inputs)):
        if type(inputs[input]) == bool:
            if inputs[input]:
                inputs[input] = 1
            else:
                inputs[input] = 0

    #        inputs = snake.get_board()
    outputs = individual.run(inputs)
    choice = outputs.index(max(outputs))
    # if outputs.count(0) == 4:
    #    choice = random.randint(0, 4)
    if choice == 1:
        snake.changeDirectionRight()
    elif choice == 0:
        snake.changeDirectionLeft()
    elif choice == 3:
        snake.changeDirectionDown()
    else:
        snake.changeDirectionUp()
    return choice


def get_output(individual, snake, novelty=False):
    #inputs = snake.get_board_rel(snake.XSIZE) + snake.find_angle_to_food()
    #inputs += snake.get_direction()
    #inputs += [(snake.body[0][0] / 7) - 1, (snake.body[0][1] / 7) - 1]
    #inputs += [(snake.food[0][0] / 7) - 1, (snake.food[0][1] /7) - 1]
    #for input in range(len(inputs)):
    #    if type(inputs[input]) == bool:
    #        if inputs[input]:
    #            inputs[input] = 1
    #        else:
    #            inputs[input] = 0
    inputs = snake.get_board_rel_2d(snake.XSIZE)
    outputs = individual.run(inputs)
    choice = outputs.index(max(outputs))
    if snake.direction == 0 and choice == 1:
        snake.direction = 0
    elif snake.direction == 1 and choice == 0:
        snake.direction = 1
    elif snake.direction == 2 and choice == 3:
        snake.direction = 2
    elif snake.direction == 3 and choice == 2:
        snake.direction = 3
    else:
        snake.direction = choice
    return choice


# This function places a food item in the environment
def placeFood(snake):
    food = []
    while len(food) < NFOOD:
        potentialfood = [random.randint(1, (snake.YSIZE - 2)), random.randint(1, (snake.XSIZE - 2))]
        if not (potentialfood in snake.body) and not (potentialfood in food):
            food.append(potentialfood)
    snake.food = food  # let the snake know where the food is
    return (food)


#snakes = [SnakePlayer(i, i) for i in range(4, 20, 2)]
snake = SnakePlayer(14,14)

def get_box(xcor, ycor):
    return xcor * 20 + 4, ycor * 20 + 4, xcor * 20 + 20, ycor * 20 + 20


def player(queue):
    global snake
    snake_i = dc(snake)
    root = tk.Tk()
    root.geometry("284x284")
    canvas = tk.Canvas(root, width=284, height=284)
    canvas.configure(background="black")
    canvas.pack()
    keep_going = True
    #root.update_idle_tasks()
    root.update()
    first = True
    while keep_going:
        keep_going, network, choice = queue.get()
        network = dc(network)
        canvas.delete("all")

        snake_i = dc(snake)
        food = placeFood(snake_i)

        for f in food:
            food_id = canvas.create_rectangle(*get_box(f[0], f[1]), fill="red")

        timer = 0
        collided = False
        body = []
        while not collided and not timer == ((2 * snake_i.XSIZE) * snake_i.YSIZE):
            root.update()
            get_output(network, snake_i)
            snake_i.updatePosition()
            if snake_i.body[0] in food:
                snake_i.score += 1
                for f in food: canvas.delete(food_id)
                food = placeFood(snake_i)
                for f in food: canvas.create_rectangle(*get_box(f[0], f[1]), fill="red")
                timer = 0
            else:
                if len(snake_i.body) > 10:
                    snake_i.body.pop()
                    canvas.delete(body[0])
                    body = body[1:]
                timer += 1  # timesteps since last eaten
            # win.addch(snake.XSIZE-2, snake.YSIZE-2, 'o')
            body.append(canvas.create_rectangle(*get_box(snake_i.body[0][0], snake_i.body[0][1]), fill="grey"))
            collided = snake_i.snakeHasCollided()
            hitBounds = (timer == ((2 * snake_i.XSIZE) * snake_i.YSIZE))
            time.sleep(0.1)


def snakeHasCollided(self):
    self.hit = False
    if self.body[0][0] == 0 or self.body[0][0] == (snake.YSIZE - 1) or self.body[0][1] == 0 or self.body[0][1] == (
            snake.XSIZE - 1): self.hit = True
    if self.body[0] in self.body[1:]: self.hit = True
    return (self.hit)


def sense_wall_ahead(self):
    self.getAheadLocation()
    return (
            self.ahead[0] == 0 or self.ahead[0] == (snake.YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (
                snake.XSIZE - 1))


def sense_food_ahead(self):
    self.getAheadLocation()
    return self.ahead in self.food


def sense_tail_ahead(self):
    self.getAheadLocation()
    return self.ahead in self.body

    print(hitBounds)
    raw_input("Press to continue...")

    return snake.score,


# This outline function provides partial code for running the game with an evolved agent
# There is no graphical output, and it runs rapidly, making it ideal for
# you need to modify it for running your agents through the game for evaluation
# which will depend on what type of EA you have used, etc.
# Feel free to make any necessary modifications to this section.


def xor_test(individual):
    fitness = float(0)
    for i in range(100):
        input = [random.choice([0, 1]), random.choice([0, 1])]
        input_temp = list(input)
        output = individual.run(input_temp)
        if (input == [0, 0] or input == [1, 1]) and output.index(max(output)) == 0:
            fitness += 1
        elif (input == [1, 0] or input == [0, 1]) and output.index(max(output)) == 1:
            fitness += 1
    return fitness / 100,


def runGameAverage(individual, gen_no=0, fitness_type="food", snake_id=5):
    output = 0
    for i in range(10):
        output += runGame(individual, gen_no=gen_no, fitness_type=fitness_type, snake_id=snake_id)
    return output / 10


def runGameAverage0(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 0)


def runGameAverage1(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 1)


def runGameAverage2(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 2)


def runGameAverage3(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 3)


def runGameAverage4(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 4)


def runGameAverage5(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 5)


def runGameAverage6(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 6)


def runGameAverage7(individual, gen_no=0, fitness_type="food"):
    return runGameAverage(individual, gen_no, fitness_type, 7)


def runGame(individual, fitness_type="food", location="", gen_no=0, snake_id=5):
    global snake
    #snake = snakes[snake_id]
    totalScore = 0
    snake = dc(snake)

    snake._reset()
    food = placeFood(snake)
    timer = 0
    timer_cumulative = [1.0, 0.0]
    alive_timer = 0
    explored_area = []
    choices = []
    while not snake.snakeHasCollided() and not timer == snake.XSIZE * snake.YSIZE:
        choice = get_output(individual, snake)
        choices.append(choice)
        snake.updatePosition()

        if snake.body[0] not in explored_area:
            explored_area.append(list(snake.body[0]))

        if snake.body[0] in food:
            snake.score += 1
            food = placeFood(snake)
            timer_cumulative = [timer_cumulative[0] + 1, timer_cumulative[1] + timer]
            timer = 0
        else:
            if len(snake.body) > 10:
                snake.body.pop()
            timer += 1  # timesteps since last eaten
        alive_timer += 1
    if fitness_type == "food":
        return snake.score

