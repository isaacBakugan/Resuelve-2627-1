# Function a: y = sin^2(x) / x, with x between -10 and 10

import math

BOUNDS = (-10, 10)  # limits of every variable
VARIABLE_COUNT = 1


def fitness(x):
    # x = 0 gives 0/0; the limit there is 0, so that value is returned
    if x == 0:
        return 0.0
    return math.sin(x) ** 2 / x
