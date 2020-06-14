from EvolutionFramework import *
from cec2013.cec2013 import *


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    benchmarkIdx = 9
    roundNum = 1
    benchmark = CEC2013(benchmarkIdx)
    for accuracy in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        hitNum = []
        peakRatio = []
        costTime = []
        for round in range(roundNum):
            startTime = time.time()
            # ea = ExperimentalNichingEA()
            ea = FastNichingEP(benchmark=benchmark, stillIterThreshold=24, maxEval=None)
            # ea = FastNichingEP_original(benchmark=benchmark, evolutionTimes=benchmark.get_no_goptima(), maxEval=None)
            # ea = FitnessSharingFEP(benchmark=benchmark)
            # ea = ProDE_Rand1(benchmark=benchmark, maxEval=None)
            ea.run()
            costTime.append(time.time()-startTime)
            # print("time used: %d" % (time.time()-startTime))
            # count, seeds = how_many_goptima(ea.resultPopulation, ea.benchmark, ea.benchmark.get_rho())
            count, seeds = how_many_goptima(ea.resultPopulation, benchmark, accuracy)
            # print(ea.benchmark.get_info())
            # print("There exist ", count, " global optimizers.")
            # print("Global optimizers:", seeds)
            # print("Peak Ratio: %f" % (count / ea.benchmark.get_no_goptima()))
            hitNum.append(count)
            pr = count / ea.benchmark.get_no_goptima()
            peakRatio.append(pr)
            # print("*****************************************************")
            print("Round: %d\nhitNum: current: %s; %s\npeakRatio: current: %s;  %s" %
                  (round, count, getListStatics(hitNum), pr, getListStatics(peakRatio)))
            print("*****************************************************")
        print("Benchmark: %d\nAccuracy: %f\nRoundNum: %d\ncostTimeAll: %s\nhitNum: %s\npeakRatio: %s\n"
              "*****************************************************" %
              (benchmarkIdx, accuracy, roundNum, sum(costTime),
               getListStatics(hitNum), getListStatics(peakRatio)))
