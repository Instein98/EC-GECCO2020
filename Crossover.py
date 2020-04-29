import random
from Utils import *


def binomialCrossover(base, mutant, crossoverRate):
    i_rand = random.randint(0, len(base)-1)
    for i in range(len(base)):
        rand = random.random()
        if rand < crossoverRate or i == i_rand:
            continue
        else:
            mutant[i] = base[i]
    return mutant


if __name__ == '__main__':
    pass
