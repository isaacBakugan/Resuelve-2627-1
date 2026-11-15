# Function b: z = 20 + x - 10*cos(2*pi*x) + y - 10*cos(2*pi*y), with x and y between -10 and 10

import math

BOUNDS = (-10, 10)  # limits of every variable
VARIABLE_COUNT = 2


def fitness(x, y):
    return 20 + x - 10 * math.cos(2 * math.pi * x) + y - 10 * math.cos(2 * math.pi * y)
