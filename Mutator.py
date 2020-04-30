import random
from Utils import *


def DERand1(population, baseIdx, probMatrix, F):
    idx1 = rouletteWheelWithProbList(probMatrix[baseIdx])
    idx2 = rouletteWheelWithProbList(probMatrix[baseIdx])
    while idx1 == baseIdx or idx2 == baseIdx or idx1 == idx2:
        idx1 = rouletteWheelWithProbList(probMatrix[baseIdx])
        idx2 = rouletteWheelWithProbList(probMatrix[baseIdx])
    return population[baseIdx] + F * (population[idx1] - population[idx2])


def FEPMutator(individual, idx, eta, newEta, benchmark):
    dim = len(individual)
    solutionValid = False
    while not solutionValid:
        randCauchy = np.random.standard_cauchy(dim)
        mutant = individual + eta[idx] * randCauchy
        solutionValid = True
        for j in range(len(mutant)):
            if mutant[j] < benchmark.get_lbound(j) or mutant[j] > benchmark.get_ubound(j):
                solutionValid = False
                break
    # update eta
    randNormal = np.random.standard_normal()
    newEta[idx] = eta[idx] * np.exp(1/np.sqrt(2*dim) * np.array([randNormal for _ in range(dim)]) +
                                    1/np.sqrt(2*np.sqrt(dim)) * np.random.standard_normal(dim))
    return mutant

