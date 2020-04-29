import random
from Utils import *


def DERand1(population, baseIdx, probMatrix, F):
    idx1 = rouletteWheelWithProbList(probMatrix[baseIdx])
    idx2 = rouletteWheelWithProbList(probMatrix[baseIdx])
    while idx1 == baseIdx or idx2 == baseIdx or idx1 == idx2:
        idx1 = rouletteWheelWithProbList(probMatrix[baseIdx])
        idx2 = rouletteWheelWithProbList(probMatrix[baseIdx])
    return population[baseIdx] + F * (population[idx1] - population[idx2])
