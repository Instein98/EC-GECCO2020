from EvolutionFramework import *
from cec2013.cec2013 import *


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    ea = MyClusterEA()
    ea.run()
    # accuracy = 0.001
    count, seeds = how_many_goptima(ea.finalPopulation, ea.benchmark, ea.benchmark.get_rho())
    print("There exist ", count, " global optimizers.")
    print("Global optimizers:", seeds)
    print("Peak Ratio: %f" % (count / ea.benchmark.get_no_goptima()))
