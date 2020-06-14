import os

from Mutator import *
from SurvivorSelector import *
from Configuration import *
from Crossover import *
from cec2013.cec2013 import *
import time


class EA:
    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize = POPULATION_SIZE,
                 ):
        self.benchmark = benchmark
        self.maxEvaluation = benchmark.get_maxfes() if maxEval is None else maxEval
        self.populationSize = populationSize
        self.dim = benchmark.get_dimension()
        self.population = self.getInitialPopulation()
        self.fitness = None
        self.resultPopulation = None
        self.startTime = None

    def getInitialPopulation(self):
        population = np.zeros((self.populationSize, self.dim))  # population matrix
        lowerBound = np.zeros(self.dim)
        upperBound = np.zeros(self.dim)
        for i in range(self.dim):
            lowerBound[i] = self.benchmark.get_lbound(i)
            upperBound[i] = self.benchmark.get_ubound(i)
        for i in range(self.populationSize):
            population[i] = lowerBound + (upperBound - lowerBound) * np.random.rand(self.dim)
        return population

    def getFitness(self, population):
        fitness = np.zeros(len(population))
        evaluationTimes = 0
        for i in range(len(population)):
            fitness[i] = self.benchmark.evaluate(population[i])
            evaluationTimes += 1
        return fitness, evaluationTimes


class ProDE_Rand1(EA):
    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize=POPULATION_SIZE,
                 ):
        super(ProDE_Rand1, self).__init__(benchmark, maxEval, populationSize)
        self.affinityMatrix = self.initAffinityMatrix()
        self.probMatrix = self.calculateProbMatrix()
        self.F = 1

    def initAffinityMatrix(self):
        affinityMatrix = np.zeros((self.populationSize, self.populationSize))
        for i in range(self.populationSize):
            for j in range(i, self.populationSize):
                if i == j:
                    affinityMatrix[i][j] = float("inf")
                else:
                    vec1 = self.population[i]
                    vec2 = self.population[j]
                    euclideanDist = getEuclideanDistance(vec2, vec1)
                    affinityMatrix[i][j] = euclideanDist
                    affinityMatrix[j][i] = euclideanDist
        return affinityMatrix

    def updateAffinityMatrix(self, idxList):
        for idx in idxList:
            for j in range(self.populationSize):
                if idx == j:
                    self.affinityMatrix[idx][j] = float("inf")
                else:
                    vec1 = self.population[idx]
                    vec2 = self.population[j]
                    euclideanDist = getEuclideanDistance(vec2, vec1)
                    self.affinityMatrix[idx][j] = euclideanDist
                    self.affinityMatrix[j][idx] = euclideanDist

    def calculateProbMatrix(self):
        probMatrix = np.zeros((self.populationSize, self.populationSize))
        for i in range(self.populationSize):
            denominator = np.sum(self.affinityMatrix[i])
            for j in range(self.populationSize):
                if i == j:
                    probMatrix[i][j] = 0
                else:
                    numerator = self.affinityMatrix[i][j]
                    probMatrix[i][j] = (1 - numerator / denominator) / (self.populationSize-2)
        return probMatrix


    def run(self):
        iterationCount = 0
        totalEvaluationTimes = 0

        while True:

            # print message
            iterationCount += 1
            # if iterationCount % 50 == 0:
                # print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
                # print("current population:\n" + str(self.population))


            # evaluation
            if self.fitness is None:
                self.fitness, evaluationTimes = self.getFitness(self.population)
                totalEvaluationTimes += evaluationTimes
                if totalEvaluationTimes > self.maxEvaluation:
                    break

            # mutation & crossover & replacement
            replacedIdxList = []
            for i in range(len(self.population)):
                solutionValid = False
                while not solutionValid:
                    solutionValid = True
                    baseVector = self.population[i]
                    # mutation
                    mutant = DERand1(self.population, i, self.probMatrix, self.F)
                    # crossover
                    mutant = binomialCrossover(baseVector, mutant, CROSSOVER_RATE)
                    # check boundary
                    for j in range(self.dim):
                        if mutant[j] < self.benchmark.get_lbound(j) or mutant[j] > self.benchmark.get_ubound(j):
                            solutionValid = False
                            break
                # replacement
                mutantFitness = self.benchmark.evaluate(mutant)
                if mutantFitness > self.fitness[i]:
                    self.population[i] = mutant
                    self.fitness[i] = mutantFitness
                    replacedIdxList.append(i)

            # update all the things
            totalEvaluationTimes += len(self.population)
            if totalEvaluationTimes > self.maxEvaluation:
                break
            self.updateAffinityMatrix(replacedIdxList)
            self.probMatrix = self.calculateProbMatrix()
        self.resultPopulation = self.population


class FastEP(EA):
    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize=POPULATION_SIZE,
                 ):
        super(FastEP, self).__init__(benchmark, maxEval, populationSize)
        self.eta = np.ones((self.populationSize, self.dim))

    def run(self):
        iterationCount = 0
        totalEvaluationTimes = 0

        while True:

            # print message
            iterationCount += 1
            # if iterationCount % 2 == 0:
                # print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
                # print("current population:\n" + str(self.population))

            # evaluation
            if self.fitness is None:
                self.fitness, evaluationTimes = self.getFitness(self.population)
                totalEvaluationTimes += evaluationTimes
                if totalEvaluationTimes > self.maxEvaluation:
                    break

            # mutation
            # print("mutating")
            newEta = np.zeros((self.populationSize, self.dim))
            newPopulation = np.zeros((self.populationSize, self.dim))
            for i, individual in enumerate(self.population):
                mutant = FEPMutator(individual, i, self.eta, newEta, self.benchmark)
                newPopulation[i] = mutant

            # replacement
            newFitness, evaluationTimes = self.getFitness(newPopulation)
            totalEvaluationTimes += evaluationTimes
            if totalEvaluationTimes > self.maxEvaluation:
                break
            self.population = roundRobinTournament(self.population, newPopulation,
                                                   self.fitness, newFitness, self.eta, newEta)
        self.resultPopulation = self.population


class FastNichingEP(EA):

    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize=POPULATION_SIZE,
                 stillIterThreshold=10,
                 nicheRadius=None
                 ):
        self.benchmark = benchmark
        self.maxEvaluation = benchmark.get_maxfes()
        self.populationSize = populationSize  # total size
        self.dim = benchmark.get_dimension()
        self.population = None
        self.fitness = None

        # self.evolutionTimes = evolutionTimes if evolutionTimes is not None else populationSize//2
        # self.maxEvaluation //= self.evolutionTimes  # divide the totalEvaluation into groups
        self.peakIndividuals = []
        self.eta = None
        if nicheRadius is None:
            upperBoundVector = np.zeros(self.dim)
            lowerBoundVector = np.zeros(self.dim)
            for k in range(self.dim):
                upperBoundVector[k] = benchmark.get_ubound(k)
                lowerBoundVector[k] = benchmark.get_lbound(k)
            self.nicheRadius = getEuclideanDistance(upperBoundVector, lowerBoundVector) / 20
        else:
            self.nicheRadius = nicheRadius
        self.resultPopulation = np.ndarray((0, self.dim))
        self.nichePopulationSize = 1
        self.worstFitness = None
        self.stillIterThreshold = stillIterThreshold

    def getFitness(self, population):
        # skipTimes = 0  # when all niches have been found, used to terminate
        fitness = np.zeros(len(population))
        evaluationTimes = 0
        for i in range(len(population)):
            inFoundNiche = False
            for j in range(len(self.resultPopulation)):
                if getEuclideanDistance(population[i], self.resultPopulation[j]) < self.nicheRadius:
                    fitness[i] = self.worstFitness
                    inFoundNiche = True
                    evaluationTimes += 1
                    # skipTimes += 1
                    break
            if not inFoundNiche:
                fitness[i] = self.benchmark.evaluate(population[i])
                evaluationTimes += 1
        return fitness, evaluationTimes

    def run(self):
        evolutionTimes = 0
        totalEvaluationTimes = 0
        while totalEvaluationTimes < self.maxEvaluation:
            evolutionTimes+=1
            # print("*"*15 + " Finding Niche " + str(x+1) + " " + "*"*15)

            self.population = self.getInitialPopulation()
            self.fitness = None
            self.eta = np.ones((self.populationSize, self.dim))
            iterationCount = 0
            optimaSameIter = 0
            lastOptima = None
            lastOptimaIdx = None

            while True:

                # print message
                iterationCount += 1
                # if iterationCount % 10 == 0:
                    # print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
                    # print("current population:\n" + str(self.population))

                # evaluation
                if self.fitness is None:
                    self.fitness, evaluationTimes = self.getFitness(self.population)
                    if totalEvaluationTimes == 0:
                        self.worstFitness = np.min(self.fitness)
                    totalEvaluationTimes += evaluationTimes
                    if totalEvaluationTimes > self.maxEvaluation:
                        break

                # mutation
                newEta = np.zeros((self.populationSize, self.dim))
                newPopulation = np.zeros((self.populationSize, self.dim))
                for i, individual in enumerate(self.population):
                    mutant = FEPMutator(individual, i, self.eta, newEta, self.benchmark)
                    newPopulation[i] = mutant

                # replacement
                newFitness, evaluationTimes = self.getFitness(newPopulation)
                totalEvaluationTimes += evaluationTimes
                if totalEvaluationTimes > self.maxEvaluation:
                    break
                self.population = roundRobinTournament(self.population, newPopulation,
                                                       self.fitness, newFitness, self.eta, newEta)

                # check #Iteration can not find better solution
                currentOptima = np.max(self.fitness)
                if lastOptima is None:
                    lastOptima = currentOptima
                else:
                    if currentOptima - lastOptima < 0.00001:
                        optimaSameIter += 1
                        lastOptima = currentOptima
                        if optimaSameIter > self.stillIterThreshold:
                            break
                    else:
                        optimaSameIter = 0
                        lastOptima = currentOptima

            print("optimaSameIter: %d" % optimaSameIter)
            bestIdx = np.argmax(self.fitness, axis=0)
            self.resultPopulation = np.concatenate((self.resultPopulation,
                                                    self.population[bestIdx:bestIdx+1]))
            # print("current niche population: " + str(self.population[:self.nichePopulationSize]))
            # print("self.resultPopulation: " + str(self.resultPopulation))
        print("evolutionTimes: %d" % evolutionTimes)


class FitnessSharingFEP(EA):

    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize=POPULATION_SIZE,
                 ):
        super(FitnessSharingFEP, self).__init__(benchmark, maxEval, populationSize)
        self.eta = np.ones((self.populationSize, self.dim))
        self.alpha = 1
        self.shareDist = 1
        self.beta = 2

    def getSharedFitness(self, population, fitness, newPopulation, newFitness):
        populationAll = np.concatenate((population, newPopulation))
        fitnessAll = np.concatenate((fitness, newFitness))
        for i, individual_i in enumerate(populationAll):
            sh_sum = 0
            for j, individual_j in enumerate(populationAll):
                # if i != j:
                dist = getEuclideanDistance(individual_i, individual_j)
                sh_ij = 1 - (dist/self.shareDist)**self.alpha if dist < self.shareDist else 0
                sh_sum += sh_ij
            fitnessAll[i] /= sh_sum
            # if i < len(fitness):
            #     fitness[i] /= sh_sum
            # else:
            #     newFitness[i-len(fitness)] /= sh_sum
        fitness = fitnessAll[:len(fitness)]
        newFitness = fitnessAll[len(fitness):]
        return fitness, newFitness

    def run(self):
        iterationCount = 0
        totalEvaluationTimes = 0

        while True:

            # print message
            iterationCount += 1
            # if iterationCount % 2 == 0:
            #     print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
            #     print("current population:\n" + str(self.population))

            # evaluation
            if self.fitness is None:
                self.fitness, evaluationTimes = self.getFitness(self.population)
                totalEvaluationTimes += evaluationTimes
                if totalEvaluationTimes > self.maxEvaluation:
                    break

            # mutation
            newEta = np.zeros((self.populationSize, self.dim))
            newPopulation = np.zeros((self.populationSize, self.dim))
            for i, individual in enumerate(self.population):
                mutant = FEPMutator(individual, i, self.eta, newEta, self.benchmark)
                newPopulation[i] = mutant

            # replacement
            newFitness, evaluationTimes = self.getFitness(newPopulation)
            totalEvaluationTimes += evaluationTimes
            if totalEvaluationTimes > self.maxEvaluation:
                break
            fitness, newFitness = self.getSharedFitness(self.population, self.fitness, newPopulation, newFitness)
            self.population = roundRobinTournament(self.population, newPopulation,
                                                   self.fitness, newFitness, self.eta, newEta)
        self.resultPopulation = self.population


# used to get entry files
class FastNichingEP_record(EA):

    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize=POPULATION_SIZE,
                 evolutionTimes=None,
                 nicheRadius=None
                 ):
        self.benchmark = benchmark
        self.maxEvaluation = benchmark.get_maxfes() if maxEval is None else maxEval
        self.populationSize = populationSize  # total size
        self.dim = benchmark.get_dimension()
        self.population = None
        self.fitness = None

        self.evolutionTimes = evolutionTimes if evolutionTimes is not None else populationSize//2
        self.maxEvaluation //= self.evolutionTimes  # divide the totalEvaluation into groups
        self.peakIndividuals = []
        self.eta = None
        if nicheRadius is None:
            upperBoundVector = np.zeros(self.dim)
            lowerBoundVector = np.zeros(self.dim)
            for k in range(self.dim):
                upperBoundVector[k] = benchmark.get_ubound(k)
                lowerBoundVector[k] = benchmark.get_lbound(k)
            self.nicheRadius = getEuclideanDistance(upperBoundVector, lowerBoundVector) / 20
        else:
            self.nicheRadius = nicheRadius
        self.resultPopulation = np.ndarray((0, self.dim))
        self.nichePopulationSize = 1
        self.worstFitness = None
        self.archive = []
        self.currentPopulationInfo = []
        self.newPopulationInfo = []  # (vector = fitness @ fes time)

    def getFitness(self, population):
        # skipTimes = 0  # when all niches have been found, used to terminate
        fitness = np.zeros(len(population))
        evaluationTimes = 0
        for i in range(len(population)):
            inFoundNiche = False
            for j in range(len(self.resultPopulation)):
                if getEuclideanDistance(population[i], self.resultPopulation[j]) < self.nicheRadius:
                    fitness[i] = self.worstFitness
                    inFoundNiche = True
                    evaluationTimes += 1
                    # skipTimes += 1
                    break
            if not inFoundNiche:
                fitness[i] = self.benchmark.evaluate(population[i])
                evaluationTimes += 1
        return fitness, evaluationTimes

    def run(self, saveFileName):
        if not os.path.exists(os.getcwd() + r"/Result"):
            os.mkdir(os.getcwd() + r"/Result")
        filePath = os.getcwd() + "/Result/" + saveFileName
        open(filePath, 'w').close()
        with open(filePath, 'a') as f:
            self.startTime = time.time()

            overalEvaluation = 0
            for x in range(self.evolutionTimes):

                self.population = self.getInitialPopulation()
                self.fitness = None
                self.eta = np.ones((self.populationSize, self.dim))
                iterationCount = 0
                totalEvaluationTimes = 0

                while True:

                    iterationCount += 1

                    # evaluation
                    if self.fitness is None:
                        self.fitness, evaluationTimes = self.getFitness(self.population)
                        if x == 0:
                            self.worstFitness = np.min(self.fitness)
                        totalEvaluationTimes += evaluationTimes
                        if totalEvaluationTimes > self.maxEvaluation:
                            break
                        else:
                            overalEvaluation += evaluationTimes

                    # mutation
                    newEta = np.zeros((self.populationSize, self.dim))
                    newPopulation = np.zeros((self.populationSize, self.dim))
                    for i, individual in enumerate(self.population):
                        mutant = FEPMutator(individual, i, self.eta, newEta, self.benchmark)
                        newPopulation[i] = mutant

                    # replacement
                    newFitness, evaluationTimes = self.getFitness(newPopulation)
                    totalEvaluationTimes += evaluationTimes
                    if totalEvaluationTimes > self.maxEvaluation:
                        break
                    else:
                        overalEvaluation += evaluationTimes
                    self.population = roundRobinTournament(self.population, newPopulation,
                                                           self.fitness, newFitness, self.eta, newEta)
                bestIdx = np.argmax(self.fitness, axis=0)
                self.resultPopulation = np.concatenate((self.resultPopulation,
                                                        self.population[bestIdx:bestIdx + 1]))
                # x1	x2	...	xd	=	y1	@	n	t	a
                foundIndividual = self.population[bestIdx]
                res = ""
                for xi in foundIndividual:
                    res += str(xi) + " "
                res += "= "
                res += str(self.fitness[bestIdx]) + " @ " + str(overalEvaluation) + \
                    " " + str(time.time()-self.startTime) + " 1\n"
                f.write(res)
                # self.resultPopulation = np.concatenate((self.resultPopulation, self.population[:self.nichePopulationSize]))
                # print("current niche population: " + str(self.population[:self.nichePopulationSize]))
                # print("self.resultPopulation: " + str(self.resultPopulation))


def record():
    for i in range(1, 21):
        benchmark = CEC2013(i)
        for roundNum in range(1, 51):
            ea = FastNichingEP_record(benchmark=benchmark, evolutionTimes=benchmark.get_no_goptima(), maxEval=None)
            # problem001run001.dat
            ea.run("problem%.3drun%.3d.dat" % (i, roundNum))



class FastNichingEP_original(EA):

    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=None,
                 populationSize=POPULATION_SIZE,
                 evolutionTimes=None,
                 nicheRadius=None
                 ):
        self.benchmark = benchmark
        self.maxEvaluation = benchmark.get_maxfes() if maxEval is None else maxEval
        self.populationSize = populationSize  # total size
        self.dim = benchmark.get_dimension()
        self.population = None
        self.fitness = None

        self.evolutionTimes = evolutionTimes if evolutionTimes is not None else populationSize//2
        self.maxEvaluation //= self.evolutionTimes  # divide the totalEvaluation into groups
        self.peakIndividuals = []
        self.eta = None
        if nicheRadius is None:
            upperBoundVector = np.zeros(self.dim)
            lowerBoundVector = np.zeros(self.dim)
            for k in range(self.dim):
                upperBoundVector[k] = benchmark.get_ubound(k)
                lowerBoundVector[k] = benchmark.get_lbound(k)
            self.nicheRadius = getEuclideanDistance(upperBoundVector, lowerBoundVector) / 20
        else:
            self.nicheRadius = nicheRadius
        self.resultPopulation = np.ndarray((0, self.dim))
        self.nichePopulationSize = 1
        self.worstFitness = None

    def getFitness(self, population):
        # skipTimes = 0  # when all niches have been found, used to terminate
        fitness = np.zeros(len(population))
        evaluationTimes = 0
        for i in range(len(population)):
            inFoundNiche = False
            for j in range(len(self.resultPopulation)):
                if getEuclideanDistance(population[i], self.resultPopulation[j]) < self.nicheRadius:
                    fitness[i] = self.worstFitness
                    inFoundNiche = True
                    evaluationTimes += 1
                    # skipTimes += 1
                    break
            if not inFoundNiche:
                fitness[i] = self.benchmark.evaluate(population[i])
                evaluationTimes += 1
        return fitness, evaluationTimes

    def run(self):
        optimaSameIterList = []
        for x in range(self.evolutionTimes):
            # print("*"*15 + " Finding Niche " + str(x+1) + " " + "*"*15)

            self.population = self.getInitialPopulation()
            self.fitness = None
            self.eta = np.ones((self.populationSize, self.dim))
            iterationCount = 0
            totalEvaluationTimes = 0
            optimaSameIter = 0
            lastOptima = None
            lastOptimaIdx = None

            while True:

                # print message
                iterationCount += 1
                # if iterationCount % 10 == 0:
                #     print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
                    # print("current population:\n" + str(self.population))

                # evaluation
                if self.fitness is None:
                    self.fitness, evaluationTimes = self.getFitness(self.population)
                    if x == 0:
                        self.worstFitness = np.min(self.fitness)
                    totalEvaluationTimes += evaluationTimes
                    if totalEvaluationTimes > self.maxEvaluation:
                        break

                # mutation
                newEta = np.zeros((self.populationSize, self.dim))
                newPopulation = np.zeros((self.populationSize, self.dim))
                for i, individual in enumerate(self.population):
                    mutant = FEPMutator(individual, i, self.eta, newEta, self.benchmark)
                    newPopulation[i] = mutant

                # replacement
                newFitness, evaluationTimes = self.getFitness(newPopulation)
                totalEvaluationTimes += evaluationTimes
                if totalEvaluationTimes > self.maxEvaluation:
                    break
                self.population = roundRobinTournament(self.population, newPopulation,
                                                       self.fitness, newFitness, self.eta, newEta)

                # check #Iteration can not find better solution
                currentOptima = np.max(self.fitness)
                if lastOptima is None:
                    lastOptima = currentOptima
                else:
                    if currentOptima - lastOptima < 0.00001:
                        optimaSameIter += 1
                        lastOptima = currentOptima
                    else:
                        optimaSameIter = 0
                        lastOptima = currentOptima

            print("optimaSameIter: %d" % optimaSameIter)
            optimaSameIterList.append(optimaSameIter)
            self.resultPopulation = np.concatenate((self.resultPopulation, self.population[:self.nichePopulationSize]))
            # print("current niche population: " + str(self.population[:self.nichePopulationSize]))
            # print("self.resultPopulation: " + str(self.resultPopulation))
        print("Average optimaSameIter: %f" % (sum(optimaSameIterList)/len(optimaSameIterList)))

if __name__ == '__main__':
    record()