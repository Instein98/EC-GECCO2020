from EvolutionFramework import *
from cec2013.cec2013 import *


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    # ea = ExperimentalNichingEA()
    ea = FastNichingEP(benchmark=CEC2013(10))
    ea.run()
    # accuracy = 0.001
    count, seeds = how_many_goptima(ea.resultPopulation, ea.benchmark, ea.benchmark.get_rho())
    print(ea.benchmark.get_info())
    print("There exist ", count, " global optimizers.")
    print("Global optimizers:", seeds)
    print("Peak Ratio: %f" % (count / ea.benchmark.get_no_goptima()))
