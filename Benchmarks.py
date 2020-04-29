import Utils

# tspFilePath = "dj38.tsp"
# DATA = []


def TSPFitness(individual, data):
    fitness = 0
    vector = individual.vector  # vector from 1!
    for i in range(len(vector)-1):
        fitness += Utils.getDistanceByCoordinate(data[vector[i] - 1], data[vector[i + 1] - 1])
    return fitness