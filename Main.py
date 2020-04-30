from EvolutionFramework import *
from cec2013.cec2013 import *


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # ea = ExperimentalNichingEA()
    ea = FastNichingEP(benchmark=CEC2013(7), evolutionTimes=40, maxEval=None)
    # ea = FitnessSharingFEP(benchmark=CEC2013(2))
    # ea = ProDE_Rand1(benchmark=CEC2013(7), maxEval=None)
    ea.run()
    # accuracy = 0.001
    count, seeds = how_many_goptima(ea.resultPopulation, ea.benchmark, ea.benchmark.get_rho())
    print(ea.benchmark.get_info())
    print("There exist ", count, " global optimizers.")
    print("Global optimizers:", seeds)
    print("Peak Ratio: %f" % (count / ea.benchmark.get_no_goptima()))
