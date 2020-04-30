import random
from Utils import *


def DERand1(population, baseIdx, probMatrix, F):
    idx1 = rouletteWheelWithProbList(probMatrix[baseIdx])
    idx2 = rouletteWheelWithProbList(probMatrix[baseIdx])
    while idx1 == baseIdx or idx2 == baseIdx or idx1 == idx2:
        idx1 = rouletteWheelWithProbList(probMatrix[baseIdx])
        idx2 = rouletteWheelWithProbList(probMatrix[baseIdx])
    return population[baseIdx] + F * (population[idx1] - population[idx2])


# todo: bitwise recalculate
def FEPMutator(individual, idx, eta, newEta, benchmark):
    maxTryTimes = 10
    tryTimes = 0
    dim = len(individual)
    mutant = np.zeros(dim)
    for i in range(dim):
        solutionValid = False
        while not solutionValid:
            tryTimes += 1
            if tryTimes > maxTryTimes:
                # mutant[i] = individual[i]  # keep consistent
                # mutant[i] = benchmark.get_lbound(i) + random.random() * \
                #     (benchmark.get_ubound(i) - benchmark.get_lbound(i))  # set random
                mutant[i] = benchmark.get_lbound(i) \
                    if mutant[i] < benchmark.get_lbound(i) else benchmark.get_ubound(i)  # set bound
                break
            randCauchy = np.random.standard_cauchy()
            mutant[i] = individual[i] + eta[idx][i] * randCauchy
            solutionValid = True
            if mutant[i] < benchmark.get_lbound(i) or mutant[i] > benchmark.get_ubound(i):
                solutionValid = False
    # update eta
    randNormal = np.random.standard_normal()
    newEta[idx] = eta[idx] * np.exp(1/np.sqrt(2*dim) * np.array([randNormal for _ in range(dim)]) +
                                    1/np.sqrt(2*np.sqrt(dim)) * np.random.standard_normal(dim))
    return mutant

