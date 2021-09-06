import math

def ramp(x): return 1 - 2 *(x%1)

def step(x):
    if (x - x % 1) % 2 == 0:
        return 1
    else:
        return -1

def spike(x):
    if (x - x % 1) % 2 == 0:
        return 1 - 2 *(x%1)
    else:
        return -1 + 2 *(x%1)

def sigmoid_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 1.0 / (1.0 + math.exp(-z))

def tanh_activation(z):
    z = max(-60.0, min(60.0, 2.5 * z))
    return math.tanh(z)

def sin_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return math.sin(z)

def gauss_activation(z):
    z = max(-3.4, min(3.4, z))
    return math.exp(-5.0 * z**2)

def relu_activation(z):
    return z if z > 0.0 else 0.0

def softplus_activation(z):
    z = max(-60.0, min(60.0, 5.0 * z))
    return 0.2 * math.log(1 + math.exp(z))

def identity_activation(z):
    return z

def clamped_activation(z):
    return max(-1.0, min(1.0, z))

def inv_activation(z):
    try:
        z = 1.0 / z
    except ArithmeticError: # handle overflows
        return 0.0
    else:
        return z

def log_activation(z):
    z = max(1e-7, z)
    return math.log(z)

def exp_activation(z):
    z = max(-60.0, min(60.0, z))
    return math.exp(z)

def abs_activation(z):
    return abs(z)

def hat_activation(z):
    return max(0.0, 1 - abs(z))

def square_activation(z):
    return z ** 2

def cube_activation(z):
    return z ** 3

