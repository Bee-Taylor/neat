import chess
import chess.svg
import random
from copy import deepcopy as dc
from svglib.svglib import svg2rlg
from cairosvg import svg2png
from reportlab.graphics import renderPM
import tkinter as tk
from PIL import Image, ImageTk
import io

class RandomAgent:
    def run(self, input):
        return random.randint(-10,10)

    def learn(self, inputs):
        return

agent_history = [RandomAgent() for i in range(10)]
roots = [None]
def board_to_input(board):
    agents_input = []
    for char in str(board):
        if char == "r":
            agents_input.append(-0.6)
        elif char == "n":
            agents_input.append(-0.3)
        elif char == "b":
            agents_input.append(-0.5)
        elif char == "q":
            agents_input.append(-0.8)
        elif char == "k":
            agents_input.append(-1.0)
        elif char == "p":
            agents_input.append(-0.1)
        elif char == "R":
            agents_input.append(0.6)
        elif char == "N":
            agents_input.append(0.3)
        elif char == "B":
            agents_input.append(0.5)
        elif char == "Q":
            agents_input.append(0.8)
        elif char == "K":
            agents_input.append(1.0)
        elif char == "P":
            agents_input.append(0.1)
        elif char == ".":
            agents_input.append(0)
    return agents_input

def evaluate_agent_pair(agent1, agent2, render = False):
    board = chess.Board()
    turn_timer = 0
    if render:
        global roots
        #root = tk.Tk()
        if roots[0] is None:
            roots[0] = tk.Tk()
        root = roots[0]
        root.geometry("390x390")
        canvas = tk.Canvas(root, width=390, height=390)
        canvas.configure(background="#aaa")
        canvas.pack()
        root.update()
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            tmp = board
            agent = agent1
            flag = False
        else:
            tmp = board.mirror()
            agent = agent2
            flag = True
        best_move = None
        best_move_value = None
        for move in tmp.legal_moves:
            tmp.push(move)
            val = agent.run(board_to_input(tmp))
            if best_move_value is None or val > best_move_value:
                best_move_value = val
                best_move = move
            tmp.pop()
        tmp.push(best_move)
        if flag:
            board = tmp.mirror()
        if turn_timer >= 200:
            break
        turn_timer += 1
        if render:
            # BAD CODE
            # todo: make not bad
            tmp_svg = chess.svg.board(board)
            img = svg2png(bytestring=tmp_svg)
            img1 = Image.open(io.BytesIO(img))
            pimg = ImageTk.PhotoImage(img1)
            canvas.pack()
            canvas.create_image(0, 0, anchor = "nw", image = pimg)
            root.update()
    result = board.result()
    if result == "1-0":
        return 1
    elif result == "0-1":
        return -1
    else:
        return 0



def evaluate_agent_pair_learning(agent1, agent2, render = False):
    board = chess.Board()
    turn_timer = 0
    if render:
        global roots
        #root = tk.Tk()
        if roots[0] is None:
            roots[0] = tk.Tk()
        root = roots[0]
        root.geometry("390x390")
        canvas = tk.Canvas(root, width=390, height=390)
        canvas.configure(background="#aaa")
        canvas.pack()
        root.update()
    cum_reward = 0
    for i in range(5):
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                tmp = board
                agent = agent1
                flag = False
            else:
                tmp = board.mirror()
                agent = agent2
                flag = True
            best_move = None
            best_move_value = None
            for move in tmp.legal_moves:
                tmp.push(move)
                val = agent.run(board_to_input(tmp))
                if best_move_value is None or val > best_move_value:
                    best_move_value = val
                    best_move = move
                tmp.pop()
            tmp.push(best_move)
            if flag:
                board = tmp.mirror()
            if turn_timer >= 200:
                break
            turn_timer += 1
            if render:
                # BAD CODE
                # todo: make not bad
                tmp_svg = chess.svg.board(board)
                img = svg2png(bytestring=tmp_svg)
                img1 = Image.open(io.BytesIO(img))
                pimg = ImageTk.PhotoImage(img1)
                canvas.pack()
                canvas.create_image(0, 0, anchor = "nw", image = pimg)
                root.update()
        result = board.result()
        if result == "1-0":
            cum_reward += 1
        elif result == "0-1":
            cum_reward += -1
        else:
            cum_reward += 0
        agent1.learn(None)
        agent2.learn(None)
    return cum_reward


def evaluate_agent(agent):
    global agent_history
    agent_history.append(dc(agent))
    if len(agent_history) > 100:
        agent_history = agent_history[2:]
    opponents = random.sample(agent_history, 5)
    cum_reward = 0
    for opponent in opponents:
        cum_reward += evaluate_agent_pair(dc(agent), dc(opponent))
    return cum_reward


def evaluate_learning_agent(agent):
    global agent_history
    agent_history.append(dc(agent))
    if len(agent_history) > 100:
        agent_history = agent_history[2:]
    opponents = random.sample(agent_history, 5)
    cum_reward = 0
    for opponent in opponents:
        cum_reward += evaluate_agent_pair_learning(dc(agent), dc(opponent))
    return cum_reward


def show_agent(agent):
    global agent_history
    evaluate_agent_pair(dc(agent), dc(random.choice(agent_history)), render = True)


def player(queue):
    keep_going = True
    first = True
    while keep_going:
        if not queue.empty() or first:
            keep_going, network, choice = queue.get()
        first = False
        show_agent(network)

