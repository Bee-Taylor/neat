import math

import numpy as np
import random

from numba import cuda

'''
ACTIVATION FUNCTIONS (IN ORDER):

OUTPUT: NO OP, RETURNS INPUT 1, BUT CAN'T BE MUTATED, MEANS THE NODE IS AN OUTPUT NODE
ITERATION: RETURNS CURRENT ITERATION INDEX
ADD: ADDS THE TWO INPUTS
SUB: SUBTRACTS THE TWO INPUTS
MULT: MULTIPLIES THE TWO INPUTS
DIV: DIVIDES INPUT 1 BY INPUT 2
ADD CONST: ADDS A CONSTANT TO INPUT 1 (UNIQUE TO NODE)
MULT CONST: MULTIPLIES INPUT 1 BY CONSTANT
SUB CONST: INPUT 1 - CONST
DIV CONST: INPUT 1 / CONST
SQRT: ROOT(INPUT 1)
POW: INPUT 1 ** INPUT 2
SQUARE: INPUT 1 ** 2
COS: COS(INPUT 1)
SIN: SIN(INPUT 1)
NOP: NO OP
CONST: RETURNS CONSTANT 
ABS: |INPUT 1|
MIN: MIN(INPUT 1, INPUT 2)
MAX: MAX(INPUT 1, INPUT 2)
CEIL: ROUNDS UP INPUT 1 (TO NEAREST 0.1)
FLOOR: ROUNDS DOWN INPUT 1
FRAC: INPUT 1 % 0.1
LOG2: LOG BASE 2 INPUT 1
RECIPRICAL: 1/INPUT1
RSQRT: 1/ROOT(INPUT1)

NETWORK STRUCTURE:
Nodes: a numpy array of shape (number of nodes + out count, 4) type: float
one node: [operation index, input 1 index, input 2 index, const]

'''


class CGP:
    def __init__(self, in_count, out_count, node_count):
        self.nodes = np.zeros((node_count + out_count, 4))
        self.node_count = node_count
        self.in_count = in_count
        self.out_count = out_count
        for i in range(in_count, node_count + in_count):
            self.nodes[i-in_count][0] = random.randint(1, 25)
            self.nodes[i-in_count][1] = random.randint(0, i-1)
            self.nodes[i-in_count][2] = random.randint(0, i-1)
            self.nodes[i-in_count][3] = random.gauss(0, 0.3333)

        for i in range(node_count, node_count + out_count):
            self.nodes[i][1] = random.randint(0, node_count + in_count)

    def run(self, inputs):
        node_vals = np.zeros(self.node_count + self.in_count + self.out_count)
        POS = 0
        for input in range(self.in_count):
            node_vals[input] = inputs[input]
        for node in range(self.node_count + self.out_count):
            if self.nodes[node][0] == 0:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])]
            elif self.nodes[node][0] == 1:
                node_vals[node + self.in_count] = POS / 105
            elif self.nodes[node][0] == 2:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] + node_vals[int(self.nodes[node][2])]
            elif self.nodes[node][0] == 3:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] - node_vals[int(self.nodes[node][2])]
            elif self.nodes[node][0] == 4:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] * node_vals[int(self.nodes[node][2])]
            elif self.nodes[node][0] == 5:
                if node_vals[int(self.nodes[node][2])] <= 0.1 and node_vals[int(self.nodes[node][2])] >= -0.1:
                    node_vals[node + self.in_count] = 1
                else:
                    node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] / node_vals[int(self.nodes[node][2])]
            elif self.nodes[node][0] == 6:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] + self.nodes[node][3]
            elif self.nodes[node][0] == 7:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] * self.nodes[node][3]
            elif self.nodes[node][0] == 8:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] - self.nodes[node][3]
            elif self.nodes[node][0] == 9:
                if self.nodes[node][3] <= 0.1 and self.nodes[node][3] >= -0.1:
                    node_vals[node + self.in_count] = 1
                else:
                    node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] / self.nodes[node][3]
            elif self.nodes[node][0] == 10:
                if node_vals[int(self.nodes[node][1])] > 0:
                    node_vals[node + self.in_count] = math.sqrt(node_vals[int(self.nodes[node][1])])
            elif self.nodes[node][0] == 11:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] ** node_vals[int(self.nodes[node][2])]
            elif self.nodes[node][0] == 12:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] ** 2
            elif self.nodes[node][0] == 13:
                node_vals[node + self.in_count] = math.cos(node_vals[int(self.nodes[node][1])])
            elif self.nodes[node][0] == 14:
                node_vals[node + self.in_count] = math.sin(node_vals[int(self.nodes[node][1])])
            elif self.nodes[node][0] == 15:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])]
            elif self.nodes[node][0] == 16:
                node_vals[node + self.in_count] = self.nodes[node][3]
            elif self.nodes[node][0] == 17:
                node_vals[node + self.in_count] = math.sqrt(node_vals[int(self.nodes[node][1])] ** 2)
            elif self.nodes[node][0] == 18:
                node_vals[node + self.in_count] = min(node_vals[int(self.nodes[node][1])], node_vals[int(self.nodes[node][2])])
            elif self.nodes[node][0] == 19:
                node_vals[node + self.in_count] = max(node_vals[int(self.nodes[node][1])], node_vals[int(self.nodes[node][2])])
            elif self.nodes[node][0] == 20:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] // 0.1 + 0.1
            elif self.nodes[node][0] == 21:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] // 0.1 - 0.1
            elif self.nodes[node][0] == 22:
                node_vals[node + self.in_count] = node_vals[int(self.nodes[node][1])] % 0.1
            elif self.nodes[node][0] == 23:
                if node_vals[int(self.nodes[node][1])] > 0:
                    node_vals[node + self.in_count] = math.log2(node_vals[int(self.nodes[node][1])])
            elif self.nodes[node][0] == 24:
                if node_vals[int(self.nodes[node][1])] != 0:
                    node_vals[node + self.in_count] = 0.1 / node_vals[int(self.nodes[node][1])]
                else:
                    node_vals[node + self.in_count] = 1
            else:
                if math.sqrt(node_vals[node + self.in_count]) != 0:
                    node_vals[node + self.in_count] = 0.1 / math.sqrt(node_vals[node + self.in_count])
                else:
                    node_vals[node + self.in_count] = 1
            if node_vals[node + self.in_count] > 10:
                node_vals[node+self.in_count] = 10
            elif node_vals[node + self.in_count] < -10:
                node_vals[node + self.in_count] = -10
        node_vals /= 10
        return node_vals[-self.out_count:]

    def mutate(self):
        for node in range(self.in_count, self.node_count + self.in_count + self.out_count):
            if random.random() < 0.05 and self.nodes[node - self.in_count][0] > 0:
                self.nodes[node - self.in_count][0] = random.randint(1, 25)
            if random.random() < 0.05:
                self.nodes[node - self.in_count][1] = random.randint(0, node - 1)
            if random.random() < 0.05:
                self.nodes[node - self.in_count][2] = random.randint(0, node - 1)
            if random.random() < 0.05:
                self.nodes[node - self.in_count][3] += random.gauss(0, 0.1)

    def add_inputs(self, to_add):
        self.in_count += to_add
        for node in self.nodes:
            node[1] += to_add
            node[2] += to_add


def mutate_individual(individual):
    individual.mutate()


#removes a bunch of features, only the basic network can be run here. NO RECURRENCY
@cuda.jit
def run_network_gpu(inputs, nodes, in_count, out_count, node_count, outputs, node_vals_big):
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

    for input in range(in_count):
        node_vals_big[image][pos][input] = inputs[image][pos][input]
    for node in range(node_count + out_count):
        if nodes[node][0] == 0:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])]
        elif nodes[node][0] == 1:
            node_vals_big[image][pos][node + in_count] = pos / 105
        elif nodes[node][0] == 2:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] + node_vals_big[image][pos][int(nodes[node][2])]
        elif nodes[node][0] == 3:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] - node_vals_big[image][pos][int(nodes[node][2])]
        elif nodes[node][0] == 4:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] * node_vals_big[image][pos][int(nodes[node][2])]
        elif nodes[node][0] == 5:
            if node_vals_big[image][pos][int(nodes[node][2])] <= 0.1 and node_vals_big[image][pos][int(nodes[node][2])] >= -0.1:
                node_vals_big[image][pos][node + in_count] = 1
            else:
                node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] / node_vals_big[image][pos][
                    int(nodes[node][2])]
        elif nodes[node][0] == 6:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] + nodes[node][3]
        elif nodes[node][0] == 7:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] * nodes[node][3]
        elif nodes[node][0] == 8:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] - nodes[node][3]
        elif nodes[node][0] == 9:
            if nodes[node][3] <= 0.1 and nodes[node][3] >= -0.1:
                node_vals_big[image][pos][node + in_count] = 1
            else:
                node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] / nodes[node][3]
        elif nodes[node][0] == 10:
            if node_vals_big[image][pos][int(nodes[node][1])] > 0:
                node_vals_big[image][pos][node + in_count] = math.sqrt(node_vals_big[image][pos][int(nodes[node][1])])
        elif nodes[node][0] == 11:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] ** node_vals_big[image][pos][int(nodes[node][2])]
        elif nodes[node][0] == 12:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] ** 2
        elif nodes[node][0] == 13:
            node_vals_big[image][pos][node + in_count] = math.cos(node_vals_big[image][pos][int(nodes[node][1])])
        elif nodes[node][0] == 14:
            node_vals_big[image][pos][node + in_count] = math.sin(node_vals_big[image][pos][int(nodes[node][1])])
        elif nodes[node][0] == 15:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])]
        elif nodes[node][0] == 16:
            node_vals_big[image][pos][node + in_count] = nodes[node][3]
        elif nodes[node][0] == 17:
            node_vals_big[image][pos][node + in_count] = math.sqrt(node_vals_big[image][pos][int(nodes[node][1])] ** 2)
        elif nodes[node][0] == 18:
            node_vals_big[image][pos][node + in_count] = min(node_vals_big[image][pos][int(nodes[node][1])],
                                                  node_vals_big[image][pos][int(nodes[node][2])])
        elif nodes[node][0] == 19:
            node_vals_big[image][pos][node + in_count] = max(node_vals_big[image][pos][int(nodes[node][1])],
                                                  node_vals_big[image][pos][int(nodes[node][2])])
        elif nodes[node][0] == 20:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] // 0.1 + 0.1
        elif nodes[node][0] == 21:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] // 0.1 - 0.1
        elif nodes[node][0] == 22:
            node_vals_big[image][pos][node + in_count] = node_vals_big[image][pos][int(nodes[node][1])] % 0.1
        elif nodes[node][0] == 23:
            if node_vals_big[image][pos][int(nodes[node][1])] > 0:
                node_vals_big[image][pos][node + in_count] = math.log2(node_vals_big[image][pos][int(nodes[node][1])])
        elif nodes[node][0] == 24:
            if node_vals_big[image][pos][int(nodes[node][1])] != 0:
                node_vals_big[image][pos][node + in_count] = 0.1 / node_vals_big[image][pos][int(nodes[node][1])]
            else:
                node_vals_big[image][pos][node + in_count] = 1
        else:
            if math.sqrt(node_vals_big[image][pos][node + in_count]) != 0:
                node_vals_big[image][pos][node + in_count] = 0.1 / math.sqrt(node_vals_big[image][pos][node + in_count])
            else:
                node_vals_big[image][pos][node + in_count] = 1
        if node_vals_big[image][pos][node + in_count] > 10:
            node_vals_big[image][pos][node + in_count] = 10
        elif node_vals_big[image][pos][node + in_count] < -10:
            node_vals_big[image][pos][node + in_count] = -10

    outputs[image][pos] = node_vals_big[image][pos][-1]


if __name__ == "__main__":
    import birdEnvironment
    import conv_network_new
    cgp = CGP(9, 1, 100)
    c = conv_network_new.conv_network(224, 224, 3, 4, 4)
    f = conv_network_new.Filter(cgp, 3)
    tmp = birdEnvironment.open_images_multiple(["ANTBIRD"])
    panes = np.zeros((3, 1, *tmp.shape[3:]))
    panes[0][0] = tmp[0][0][0]
    panes[1][0] = tmp[0][1][0]
    panes[2][0] = tmp[0][2][0]
    threadsperblock = 64

    inputs = f.get_inputs(panes)[:4]

    blockspergrid = (inputs.size + (threadsperblock - 1)) // threadsperblock

    outputs = np.zeros((panes.shape[0], panes.shape[2] * panes.shape[3]), dtype=float)
    node_vals = np.zeros((inputs.shape[0], inputs.shape[1], cgp.node_count + cgp.in_count + cgp.out_count), dtype=float)

    for i in range(100):
        run_network_gpu[blockspergrid, threadsperblock](inputs, cgp.nodes, cgp.in_count, cgp.out_count, cgp.node_count, outputs, node_vals)
        outputs /= 10
        new_outputs = np.reshape(outputs, (panes.shape[0], *panes.shape[2:]))
        c.show_images(new_outputs, outputs.shape[0])
        mutate_individual(cgp)
