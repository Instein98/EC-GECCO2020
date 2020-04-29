# Measurement

### 1. $ \frac{\# DetectedItems}{\#RelevantItems}$(# Detected Peaks / Optimum Found )

> The other important change concerns **the performance criteria**. As before, we fix the maximum amount of evaluations per run and problem and count the number of detected peaks as our first measure. This is also known as **recall** in information retrieval, or sensitivity in binary classification. Recall is defined as the number of successfully detected items (out of the set of relevant items), divided by the number of relevant items.

### 2. $\frac{\#DetectedPeaks}{\#Solutions}$(F1 measure)

> Additionally, we will also look at 2 more measures, namely the static **F1 measure** (after the budget has been used up), and the (dynamic) **F1 measure integral**. The F1 measure is usually understood as the product of precision and recall. **Precision** is meant as the fraction of relevant detected items compared to the overall number of detected items, in our case this is the number of detected peaks divided by the number of solutions in the report ("best") file. That is, the higher the number of duplicates or **non-optimal solutions** in the report, the worse the computed performance indicator gets. **Ideally, the report file shall provide all sought peaks and only these, which would result in an F1 value of 1 (as both recall and precision would be 1)**.

### 3. AUC of the "current F1 value -- time" curve 

> The static F1 measure uses the whole report file and computes the F1 measures of complete runs, and these are then averaged over the number of runs and problems. The 3rd measure is more fine-grained and also looks at the time (in function evaluation counts) when solutions are detected. Therefore, track the report file line by line and compute the F1 value for each point in time when a new solution (line) is written, using all the information that is available up to that point. Thus we compute the F1 value "up to that time", and doing that for each step results in a curve. The F1 measure integral is the area under-the-curve (AUC), divided by the maximum number of function evaluations allowed for that specific problem. It is therefore also bounded by 1.

# Local Optimum

一个曲面是连续的，为什么说有n个局部最优？有两个凹陷。如果一个点不是局部最优点，那么它一定能通过一条斜率始终小于0的路径到达一个更低的点。“平的或者下坡”，“路径”，“更低的点”。“能继续下坡”

exploitative strategy：个体变异，相近个体的交叉。（希望尽可能是同峰解来交叉）

explorative strategy：个体变异。（探索具有一定随机性）

变异：变异粒度小，则用作局部搜索

交叉：如果个体相近，则为局部；距离远则全局。

如何保证全部的population能分为多个类群。

# 将DE和fitness sharing结合的优劣

### Flat Selector:

如果有些点不在全局最优，而在局部最优或者是任何一点，就有可能放大那些非最优的解。

无论 fitness sharing 和 还是DE都没有考虑上面这个问题，而仅仅只是基于距离、cluster来考虑。

### Cluster Selector - Convergence（选择压力非常重要！）:

采用概率的方式，既对适应度较低的个体产生压力，又能将种群尽量分散开来。需要一个可以退火的参数。

基于概率的排除策略，概率C是进行近距离排除策略的概率：

1. C的概率进行近距离排除策略，排除掉距离最近的两个个体中较差的一个。
2. 1-C的概率进行适应度淘汰策略，根据适应度进行Roulette Wheel的排除，排除一个个体。
3. C进行退火，C的最小值不能为0，否则不能达到niching的效果。
   - 最小值设置为0.2，意味着绝大部分的策略为淘汰劣势，从而选择出较优的cluster，后期F变小，“越聚集的地方会产生越多的个体”的问题越明显，近距离排除策略依然很重要
   - 考虑对比 0.1 0.2 0.3 0.4 0.5

优点：

- 由于DE里mutation和operator的特点，越聚集的地方会产生越多的个体，这种selector可以有效平衡这个问题。
- 在开始的时候能尽量使个体均匀分散，保证种群diversity，达到exploration的效果；后期逐渐加大选择压力，能逐渐淘汰表现较差的cluster，把个体集中到peak上。

缺点：

- 计算距离时开销比较大。因此可以考虑控制种群大小以及子代数量。

### Cluster Selector - simplified:

近距离排除时，不把新旧种群放在一起进行。

选择旧种群中距离最近的两个个体，留下较好个体，选择与较好个体最远的个体进来。

重复以上操作 种群数量/2 次。节省3倍开销。

### Cluster Selector - C=1 对比:

目标：将距离最近的个体筛选出去，确保种群的多样性

方式：从DE的$R_d$进行拓展，计算出所有的距离，选择最近的个体进行比较，留下好的那个。

缺点：开销巨大。难以保证算法的收敛性。无法将适应度较小的个体筛选出去。

预期：最后种群内个体一定会变得非常分散。但是根本找不出峰值。

优点：由于DE里mutation和operator的特点，越聚集的地方会产生越多的个体，这种selector可以有效平衡这个问题。

### Roulette Wheel Selector 对比

### fitness sharing (Selector) 对比:

设置参数：sharing radius， population size， 

### Crowding 对比

### Cluster Selector - C=0 对比:

### Pro DE 对比

### 收敛条件

连续10代没有找到更好的个体了。



### Pro DE:

基于近距离的个体产生新个体，控制F的值实现先explore后exploit。F模拟退火。

选择器的缺点：

- 选择生成的个体和原本个体更好的那个，如果生成的个体在更远的cluster，可能因此丧失可能性。Exploration比较受限。
- Exploration只在变异交叉后的个体比原来个体好时才体现出来，不能接受更差的个体。



# 思考

### Pro DE rand + binomial + rank select

为啥不能niching？