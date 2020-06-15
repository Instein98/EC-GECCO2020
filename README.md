## Run FastNichingEP (dump result file)

In main method of Main.py:

```python
for i in range(1, 21):
    benchmark = CEC2013(i)
    for roundNum in range(1, 51):
        ea = FastNichingEP_record(benchmark=benchmark)
        # problem001run001.dat
        ea.run("problem%.3drun%.3d.dat" % (i, roundNum))
```

## Run FitnessSharingFEP

```python
if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    benchmarkIdx = 1
    roundNum = 1
    benchmark = CEC2013(benchmarkIdx)
    hitNum = []
    peakRatio = []
    costTime = []
    for round in range(roundNum):
        startTime = time.time()
        ea = FitnessSharingFEP(benchmark=benchmark)
        ea.run()
        costTime.append(time.time()-startTime)
        count, seeds = how_many_goptima(ea.resultPopulation, benchmark, accuracy)
        hitNum.append(count)
        pr = count / ea.benchmark.get_no_goptima()
        peakRatio.append(pr)
        print("Round: %d\nhitNum: current: %s; %s\npeakRatio: current: %s;  %s" %
              (round, count, getListStatics(hitNum), pr, getListStatics(peakRatio)))
        print("*****************************************************")
    print("Benchmark: %d\nAccuracy: %f\nRoundNum: %d\ncostTimeAll: %s\nhitNum: %s\npeakRatio: %s\n"
              "*****************************************************" %
              (benchmarkIdx, accuracy, roundNum, sum(costTime),
               getListStatics(hitNum), getListStatics(peakRatio)))
```


## Run ProDE_Rand1

```python
if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    benchmarkIdx = 1
    roundNum = 1
    benchmark = CEC2013(benchmarkIdx)
    hitNum = []
    peakRatio = []
    costTime = []
    for round in range(roundNum):
        startTime = time.time()
        ea = ProDE_Rand1(benchmark=benchmark)
        ea.run()
        costTime.append(time.time()-startTime)
        count, seeds = how_many_goptima(ea.resultPopulation, benchmark, accuracy)
        hitNum.append(count)
        pr = count / ea.benchmark.get_no_goptima()
        peakRatio.append(pr)
        print("Round: %d\nhitNum: current: %s; %s\npeakRatio: current: %s;  %s" %
              (round, count, getListStatics(hitNum), pr, getListStatics(peakRatio)))
        print("*****************************************************")
    print("Benchmark: %d\nAccuracy: %f\nRoundNum: %d\ncostTimeAll: %s\nhitNum: %s\npeakRatio: %s\n"
              "*****************************************************" %
              (benchmarkIdx, accuracy, roundNum, sum(costTime),
               getListStatics(hitNum), getListStatics(peakRatio)))
```


## Measurement

### 1. $ \frac{\# DetectedItems}{\#RelevantItems}$(# Detected Peaks / Optimum Found )

> The other important change concerns **the performance criteria**. As before, we fix the maximum amount of evaluations per run and problem and count the number of detected peaks as our first measure. This is also known as **recall** in information retrieval, or sensitivity in binary classification. Recall is defined as the number of successfully detected items (out of the set of relevant items), divided by the number of relevant items.

### 2. $\frac{\#DetectedPeaks}{\#Solutions}$(F1 measure)

> Additionally, we will also look at 2 more measures, namely the static **F1 measure** (after the budget has been used up), and the (dynamic) **F1 measure integral**. The F1 measure is usually understood as the product of precision and recall. **Precision** is meant as the fraction of relevant detected items compared to the overall number of detected items, in our case this is the number of detected peaks divided by the number of solutions in the report ("best") file. That is, the higher the number of duplicates or **non-optimal solutions** in the report, the worse the computed performance indicator gets. **Ideally, the report file shall provide all sought peaks and only these, which would result in an F1 value of 1 (as both recall and precision would be 1)**.

### 3. AUC of the "current F1 value -- time" curve 

> The static F1 measure uses the whole report file and computes the F1 measures of complete runs, and these are then averaged over the number of runs and problems. The 3rd measure is more fine-grained and also looks at the time (in function evaluation counts) when solutions are detected. Therefore, track the report file line by line and compute the F1 value for each point in time when a new solution (line) is written, using all the information that is available up to that point. Thus we compute the F1 value "up to that time", and doing that for each step results in a curve. The F1 measure integral is the area under-the-curve (AUC), divided by the maximum number of function evaluations allowed for that specific problem. It is therefore also bounded by 1.

