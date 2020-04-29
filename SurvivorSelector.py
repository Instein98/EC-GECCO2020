import random
import numpy as np
from Utils import *


def clusterSelector(population, fitness, newPopulation, newFitness, affinityMatrix, C):
    totalSize = len(population) + len(newPopulation)
    affinityMatrixAll = np.zeros((totalSize, totalSize))
    affinityMatrixAll[0:affinityMatrix.shape[0], 0:affinityMatrix.shape[1]] = affinityMatrix
    populationAll = np.concatenate((population, newPopulation))
    fitnessAll = np.concatenate((fitness, newFitness))
    # complete affinityMatrixAll
    for i in range(len(newPopulation)):
        actual_i = len(population) + i
        for j in range(len(population)):
            vec1 = newPopulation[i]
            vec2 = population[j]
            euclideanDist = np.sqrt(np.sum((vec1 - vec2) ** 2))
            affinityMatrixAll[actual_i][j] = euclideanDist
            affinityMatrixAll[j][actual_i] = euclideanDist
        for j in range(len(newPopulation)):
            actual_j = len(population) + j
            if i == j:
                affinityMatrixAll[actual_i][actual_j] = float("inf")
            else:
                vec1 = newPopulation[i]
                vec2 = newPopulation[j]
                euclideanDist = np.sqrt(np.sum((vec1 - vec2) ** 2))
                affinityMatrixAll[actual_i][actual_j] = euclideanDist
                affinityMatrixAll[actual_j][actual_i] = euclideanDist
    # move out the individuals
    while len(populationAll) > len(population):
        rand = random.random()
        if rand < C:
            # # remove the nearest
            # tmp = np.where(affinityMatrixAll == np.amin(affinityMatrixAll))
            # candidateIdx1 = tmp[0][0]
            # candidateIdx2 = tmp[1][0]
            # removeIdx = candidateIdx2 if fitnessAll[candidateIdx1] > fitnessAll[candidateIdx2] else candidateIdx1
            # affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 0)
            # affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 1)
            # populationAll = np.delete(populationAll, removeIdx, 0)
            # fitnessAll = np.delete(fitnessAll, removeIdx, 0)

            # remove by flat
            removeIdx = random.randint(0, len(fitnessAll)-1)
            affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 0)
            affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 1)
            populationAll = np.delete(populationAll, removeIdx, 0)
            fitnessAll = np.delete(fitnessAll, removeIdx, 0)
        else:
            # # remove by rouletteWheel
            # fitnessSum = sum(fitnessAll)
            # probList = [(1 - fitnessAll[i]/fitnessSum)/(len(fitnessAll)-1) for i in range(len(fitnessAll))]
            # removeIdx = rouletteWheelWithProbList(probList)
            # affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 0)
            # affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 1)
            # populationAll = np.delete(populationAll, removeIdx, 0)
            # fitnessAll = np.delete(fitnessAll, removeIdx, 0)

            # remove by rank based selection
            individualNum = len(fitnessAll)
            tmp = [(i, fitnessAll[i]) for i in range(individualNum)]
            tmp.sort(key=lambda x: x[1], reverse=True)  # fitness descending
            probList = [(i+1)/((1+individualNum)*individualNum/2) for i in range(individualNum)]
            tmpIdx = rouletteWheelWithProbList(probList)
            while tmpIdx == 0:
                tmpIdx = rouletteWheelWithProbList(probList)
            removeIdx = tmp[tmpIdx][0]
            affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 0)
            affinityMatrixAll = np.delete(affinityMatrixAll, removeIdx, 1)
            populationAll = np.delete(populationAll, removeIdx, 0)
            fitnessAll = np.delete(fitnessAll, removeIdx, 0)

    return populationAll, fitnessAll, affinityMatrixAll

