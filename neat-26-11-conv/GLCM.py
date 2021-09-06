from numba import cuda
import numpy as np
import random


class GLCM_Filter:
    def __init__(self, displacement_vector):
        self.displacement_vector = displacement_vector

    def run(self, panes):
        threadsperblock = 64
        blockspergrid = (panes.size + (threadsperblock - 1)) // threadsperblock
        outputs = np.zeros((panes.shape[0], 4, panes.shape[2], panes.shape[3]), dtype=float)
        outputs -= 1
        glcm_gpu[blockspergrid, threadsperblock](panes, outputs, self.displacement_vector)
        return outputs

    def mutate_displacement_vector(self):
        if random.random() >= 0.5:
            if random.random() < 0.5:
                self.displacement_vector[0] -= 1
            else:
                self.displacement_vector[0] += 1
        if random.random() >= 0.5:
            if random.random() < 0.5:
                self.displacement_vector[1] -= 1
            else:
                self.displacement_vector[1] += 1


def mutate_individual(individual):
    return individual.mutate_displacement_vector()


@cuda.jit
def glcm_gpu(panes, outputs, displacement_vector):
    # Thread id in a 1D block
    tx = cuda.threadIdx.x
    # Block id in a 1D grid
    ty = cuda.blockIdx.x
    # Block width, i.e. number of threads per block
    bw = cuda.blockDim.x
    # Compute flattened index inside the array

    posBig = tx + ty * bw
    target_image = posBig // panes[0].size

    pos = posBig % panes[0].size
    p_target = pos // (panes.shape[2] * panes.shape[3])
    inner_pos = pos % (panes.shape[2] * panes.shape[3])
    y = inner_pos // panes.shape[3]
    x = inner_pos % panes.shape[3]
    if target_image >= panes.shape[0] \
            or p_target >= panes.shape[1] \
            or y + displacement_vector[0] >= panes.shape[2] \
            or x + displacement_vector[1] >= panes.shape[3]:
        return
    if panes[target_image][p_target][y][x] >= 0: 
        if panes[target_image][p_target][y+displacement_vector[0]][x+displacement_vector[1]] >= 0: 
            outputs[target_image][0][y][x] += 2 / panes.shape[1]
        else: 
            outputs[target_image][1][y][x] += 2 / panes.shape[1]
    else: 
        if panes[target_image][p_target][y+displacement_vector[0]][x+displacement_vector[1]] >= 0: 
            outputs[target_image][2][y][x] += 2 / panes.shape[1]
        else: 
            outputs[target_image][3][y][x] += 2 / panes.shape[1]


if __name__ == "__main__":
    f = GLCM_Filter(np.array([0,1], dtype=int))
    tmp = np.array([[[[1,1,0,0], [1,1,0,0], [0,0,-1,-1],[0,0,-1,-1]]]])
    print(f.run(tmp))