[TOC]



# 写在前面（对常见RNN模型的总结）

## 1.NN模型（neural network）

![1540554004821](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540554004821.png)

1.NN模型是复杂神经网络模型的基础计算单元

2.常见的激活函数主要有3种：sigmoid函数，tanh函数，以及ReLU（rectified linear unit）函数

3.NN模型若放在输出层，不同的任务使用不同的激活函数：

（1）多任务（multi-class）：使用softmax函数（即 归一化指数函数）

![img](https://img-blog.csdn.net/20150917130313199?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQv/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

（2）多标签（multi-label）：每个节点使用一个sigmoid函数

## 2. FFNN模型（feedforward neural network）

![1540554261170](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540554261170.png)



1.工作方式：通过设置最低层的值来提供前馈网络的输入x，然后连续计算每个较高层，直到在最顶层y处产生输出。

2..FFNN模型的训练的关键在于优化的损失函数（ loss function L）

3.优化最成功的算法：BP算法

原理：BP算法由信号的正向传播和误差的反向传播两个过程组成。

正向传播时，输入样本从输入层进入网络，经隐层逐层传递至输出层，如果输出层的实际输出与期望输出(导师信号)**不同** ，则转至误差反向传播；如果输出层的实际输出与期望输出(导师信号)相同，结束学习算法。

反向传播时，将输出误差(期望输出与实际输出之差)按原通路反传计算，通过隐层反向，直至输入层，在反传过程中将误差**分摊给各层的各个单元** ，获得各层各单元的误差信号，并将其作为修正各单元权值的根据。这一计算过程使用梯度下降法完成，在不停地调整各层神经元的权值和阈值后，使误差信号减小到最低限度。

权值和阈值不断调整的过程，就是网络的学习与训练过程，经过信号正向传播与误差反向传播，权值和阈值的调整反复进行，一直进行到预先设定的学习训练次数，或输出误差减小到允许的程度。

##3.RNN模型 

![1540556769287](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540556769287.png)

​										有环形式

![1540556798119](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540556798119.png)

​										无环形式

1.**隐藏层的反馈，不仅仅进入输出端，而且还进入了下一时间的隐藏层** 

2.我的理解：![1540557188265](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540557188265.png)

（假设当前层为时间t）上式左边的$h^t$是第t层的hidden note的表达式

上式右边的$W^{hx} W^{hh}$ 是$x^t$和$h^(t-1)$所对应的权重值

PS：权重（从输入到隐藏和隐藏到输出）在每个时间步是相同的（固定权重的边来使隐藏节点自我连接的想法是LSTM网络后续工作的基础）

输出方程：

![1540557258078](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540557258078.png)



3.训练RNN

考虑单个输入节点，单个输出节点和单个循环隐藏节点的网络：![1540558436168](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540558436168.png)

现在假如时间$\tau$ 有一个非零输入，然后在时间$\tau$到时间t（间隔很长）的输入都是0，那么在时间t来看时间$\tau$输入的影响将是爆炸的，或者是微小的

发生这两种现象中的哪一种取决于**循环边的权重|**  wjj | > 1或| wjj | <1 以及 隐藏节点中的**激活函数** 

**4.消失梯度问题的可视化：** 

如果沿循环边缘的权重小于1，则第一时间步的输入对最终时间步的输出的贡献将以**指数方式快速减小**

# LSTM(Long Short Term Memory networks)

##      阅读材料

1. [Understanding LSTM Networks]:http://colah.github.io/posts/2015-08-Understanding-LSTMs/

2. [Attention and Augmented Recurrent Neural Networks]:https://distill.pub/2016/augmented-rnns/?tdsourcetag=s_pcqq_aiomsg#attentional-interfaces

   ## 关键点

* 1

  + ![1540377534263](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540377534263.png)

    将RNN视为同一网络的多个副本

* 2

  LSTM能够学习长期依赖问题，而一般的RNN不行

* 3

  The sigmoid layer outputs numbers between zero and one, describing how much of each component should be let through.

  0 means "let nothings through"

  1 means "let everything through"

  An LSTM has **three** of these gates, to protect and control the cell state.

  # The Orders of LSTM

  * [ ] 1.Forget gate layer:decide what information we’re going to throw away from the cell state

  * [ ] $$
    f_t=\sigma(W_f*[h_{t-1},x_t] + b_f)
    $$

  * [ ] 2.Decide what new information we’re going to store in the cell state

    + Input gate layer:decide which values we'll update
    + A tanh layer:creates a vector of new candidate values
    + Update the old cell state

  * multiply the old state by $f_t$ ,  forgetting the things we decided to forget earlier.

    + ![1540380352361](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540380352361.png)

      and then add $\overset{\thicksim}C_t*i_t$

    + **$C_t=f_t*C_{t-1}+\overset{\thicksim}C_t*i_t$**

  * [ ] Finally, we need to decide what we’re going to output

    + out put $h_t$

    + ![1540381550132](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540381550132.png)



## 论文：A Critical Review of Recurrent Neural Networks for Sequence Learning

​    

### 论文的研究目的

+ [ ] 1.回顾和综合了过去三十年来首次产生的研究，然后使这些强大的学习模型变得实用
+ [ ] 2.现有的论文普遍符合的命名不统一，文章算是一篇综述

### 论文的内容

背景 -》RNN的前生 -》现代的RNN -》LSTM和BRNN的应用 -》讨论和总结

### 具体理解

1. 为什么是序列模型（model sequentiality）？

   我们想要学习长期依赖问题，而序列不可知模型在没有明确建模时间的情况下非常有用

   像SVM、LR、前向反馈网络是建立在 “独立”假设的基础上，更多模型则是人为地去构造前后顺序。

   但即使这样，上述模型仍然不能解决长时间序列的依赖问题。比如语音或文本识别中的长句子场景。

   解决方法：

   通常情况下是一个一个挨着输入，但文中是将t时刻的输入和t-1,t+1时刻一起输入 （根据具体情况可以更长）

   这样可能会带来更高的模型精度，但训练复杂度会增加

2. 为什么不用马尔科夫模型？

   先来了解一下背景知识：

   >**马尔可夫链**:指数学中具有马尔可夫性质的离散事件随机过程。该过程中，在给定当前知识或信息的情况下，过去（即当前以前的历史状态）对于预测将来（即当前以后的未来状态）是无关的
   >
   >![1540436719160](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540436719160.png)
   >
   >这里x为过程中的某个状态。上面这个恒等式可以被看作是**马尔可夫性质**。
   >
   >
   >
   >即$X_{n+1}$对于**过去状态**的**条件概率**分布仅是$X_n$的一个函数
   >
   >
   >
   >**隐马尔可夫模型(HMM):**马尔可夫链的一种，它的状态不能直接观察到，但能通过观测**向量序列**观察到，每个观测向量都是通过某些概率密度分布表现为各种状态，每一个观测向量是由一个具有相应概率密度分布的状态序列产生。

   从概念中可以看出，HMM中的每个隐藏状态仅取决于**紧接在前的状态**。

   虽然可以通过创建一个等于窗口中每次可能状态的叉积的新状态空间来扩展马尔可夫模型以考虑更大的上下文窗口，但此过程对于窗口而言，会使状态空间**呈指数增长**。

   所以HMM在建立**远程依赖性**的计算上是不切实际。

3. RNN会过拟合吗？

   具有非线性激活单元的有限大小的RNN几乎可以进行任何计算。但是不会过拟合。原因如下： 
   （1）RNN的每一层都是可以进行求导的，所以能对参数数量（梯度计算）有很好的控制； 
   （2）可以通过权重衰减、dropout等方法对模型进行限制。

4. RNN是前馈神经网络，通过**相邻时间的跨度**来增强，引入时间概念

5. Elman using bp 证明网络可以学习时间依赖性

   ![1540557998321](C:\Users\LL\AppData\Roaming\Typora\typora-user-images\1540557998321.png)

   输出单元连接到特殊单元，这些单元在下一步进入自我训练并训练隐藏单元。

6. 

7. 







