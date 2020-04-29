'''
Before parent selection, fitness score should be calculated in advance!
'''

import random
from Configuration import *


def tournamentSelector(population):
    candidates = []
    while len(candidates) < K:
        idx = int(random.random() * len(population))  # [0, len(population))
        while idx in candidates:
            idx = int(random.random() * len(population))  # [0, len(population))
        candidates.append(idx)
    if MAXIMIZATION:
        candidates.sort(key=lambda x: population[x].fitness, reverse=True)  # decreasing
    else:
        candidates.sort(key=lambda x: population[x].fitness, reverse=False)  # increasing
    return population[candidates[0]]


# rank 0 is the worst!!
def rankBasedSelector(population):
    rankSum = (1 + len(population)) * len(population) / 2
    if MAXIMIZATION:
        population.sort(key=lambda x: x.fitness, reverse=False)  # increasing
    else:
        population.sort(key=lambda x: x.fitness, reverse=True)  # decreasing
    for i, individual in enumerate(population):
        individual.selectProb = (i + 1) / rankSum
    return selectParentByProb(population)


def rouletteWheelSelector(population):
    fitnessSum = sum([x.fitness for x in population])
    for individual in population:
        individual.selectProb = individual.fitness / fitnessSum  # maximization
        if not MAXIMIZATION:
            individual.selectProb = (1 - individual.selectProb) / (len(population) - 1)
    return selectParentByProb(population)


def selectParentByProb(population):
    probSum = 0
    probList = []
    for individual in population:
        probSum += individual.selectProb
        probList.append(probSum)
    r = random.random()  # 0<=r<1
    for i, p in enumerate(probList):
        if r < p:
            return population[i]
    return None
