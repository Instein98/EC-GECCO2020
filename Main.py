from EvolutionFramework import *
from cec2013.cec2013 import *


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    for i in range(1, 21):
        benchmark = CEC2013(i)
        for roundNum in range(1, 51):
            ea = FastNichingEP_record(benchmark=benchmark)
            # problem001run001.dat
            ea.run("problem%.3drun%.3d.dat" % (i, roundNum))
