from Mutator import *
from SurvivorSelector import *
from Configuration import *
from Crossover import *
from cec2013.cec2013 import *


class EA:
    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=MAX_EVALUATION,
                 populationSize = POPULATION_SIZE,
                 ):
        self.benchmark = benchmark
        self.maxEvaluation = maxEval
        self.populationSize = populationSize
        self.dim = benchmark.get_dimension()
        self.population = self.getInitialPopulation()
        self.fitness = None

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


class ExperimentalNichingEA(EA):
    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=MAX_EVALUATION,
                 populationSize=POPULATION_SIZE,
                 ):
        super(ExperimentalNichingEA, self).__init__(benchmark, maxEval, populationSize)
        self.affinityMatrix = self.initAffinityMatrix()
        self.probMatrix = self.updateProbMatrix()
        self.F = F_INITIAL
        self.C = C_INITIAL
        # self.finalPopulation = None

    def initAffinityMatrix(self):
        affinityMatrix = np.zeros((self.populationSize, self.populationSize))
        for i in range(self.populationSize):
            for j in range(i, self.populationSize):
                if i == j:
                    affinityMatrix[i][j] = float("inf")
                else:
                    vec1 = self.population[i]
                    vec2 = self.population[j]
                    euclideanDist = np.sqrt(np.sum((vec1 - vec2) ** 2))
                    affinityMatrix[i][j] = euclideanDist
                    affinityMatrix[j][i] = euclideanDist
        return affinityMatrix

    def updateProbMatrix(self):
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
        evaluationRunOut = False

        while not evaluationRunOut:

            # print message
            iterationCount += 1
            if iterationCount % 50 == 0:
                print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
                print("F = %f" % self.F)
                print("C = %f" % self.C)
                print("current population:\n" + str(self.population))

            self.F = F_INITIAL - (F_INITIAL - F_LBOUND) * (totalEvaluationTimes/self.maxEvaluation)
            self.C = C_INITIAL - (C_INITIAL - C_LBOUND) * (totalEvaluationTimes/self.maxEvaluation)

            # evaluation todo: only update the changed individuals
            if self.fitness is None:
                self.fitness, evaluationTimes = self.getFitness(self.population)
                totalEvaluationTimes += evaluationTimes
                if totalEvaluationTimes > self.maxEvaluation:
                    evaluationRunOut = True
                    break

            # mutation & crossover
            newPopulation = np.zeros((LAMBDA, self.dim))
            for i in range(LAMBDA):
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
                newPopulation[i] = mutant

            # replacement
            newFitness, evaluationTimes = self.getFitness(newPopulation)
            totalEvaluationTimes += evaluationTimes
            if totalEvaluationTimes > self.maxEvaluation:
                evaluationRunOut = True
                break
            self.population, self.fitness, self.affinityMatrix = \
                clusterSelector(self.population, self.fitness, newPopulation, newFitness, self.affinityMatrix, self.C)
            self.updateProbMatrix()
        # self.finalPopulation = self.population


class FastEP(EA):
    def __init__(self,
                 benchmark=CEC2013(1),
                 maxEval=MAX_EVALUATION,
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
            if iterationCount % 2 == 0:
                print("Iteration: %d, Current Best: %f" % (iterationCount, max(self.fitness)))
                print("current population:\n" + str(self.population))

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
            self.population = roundRobinTournament(self.population, newPopulation,
                                                   self.fitness, newFitness, self.eta, newEta)
