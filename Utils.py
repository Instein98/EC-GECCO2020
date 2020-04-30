import math
import numpy as np
import random

class Individual:
    def __init__(self, vector):
        self.vector = vector
        self.fitness = None
        self.selectProb = None
        self.winTimes = 0

    def copy(self):
        return Individual(self.vector.copy())


class Benchmark:
    def __init__(self, tspFilePath):
        self.name = tspFilePath.split("/")[-1][:-4]
        self.path = tspFilePath
        self.data = self.readDataFile(tspFilePath)

    def readDataFile(self, path):
        result = []
        start = False
        with open(path, mode='r') as f:
            line = f.readline()
            while line:
                if start:
                    data = line.split()
                    result.append((float(data[1]), float(data[2])))
                if line == "NODE_COORD_SECTION\n":
                    start = True
                if line == "EOF\n":
                    break
                line = f.readline()
        return result

    def getDistanceByCityIdx(self, idx1, idx2):
        return getDistanceByCoordinate(self.data[idx1-1], self.data[idx2-1])


def getDistanceByCoordinate(tuple1, tuple2):
    deltaX = abs(tuple1[0] - tuple2[0])
    deltaY = abs(tuple1[1] - tuple2[1])
    return math.sqrt(deltaX * deltaX + deltaY * deltaY)


def getEntropy(population):
    n = len(population[0].vector)-1  # city count
    p = len(population)  # population size
    def getNij(i, j):
        res = 0
        for individual in population:
            if i == 1:
                if individual.vector.index(j) == 1 or individual.vector.index(j) == n-1:
                    res += 1
            else:
                idx = individual.vector.index(i)
                print([i, idx])
                print(individual.vector)
                if individual.vector[idx+1] == j or individual.vector[idx-1] == j:
                    res += 1
        return res
    hi_list = []
    for i in range(1, n+1):
        sum = 0
        for j in range(1, n+1):
            Nij = getNij(i, j)
            sum += 0 if Nij == 0 else Nij/(2*p) * math.log(Nij/(2*p))
        hi_list.append(-sum/(math.log(n)))
    return np.mean(hi_list)


def rouletteWheelWithProbList(probList):
    probSumList = [sum(probList[:i+2]) for i in range(len(probList))]
    rand = random.random()
    for i, p in enumerate(probSumList):
        if rand < p:
            return i


def getEuclideanDistance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))


if __name__ == '__main__':
    pass
