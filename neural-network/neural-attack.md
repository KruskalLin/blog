### 攻击概念

#### 范式

用来度量攻击样本和原样本的差距，一般公式是

![formula](https://latex.codecogs.com/gif.latex?%0A%7C%7Cx-x%27%7C%7C_p%3D%28%5Csum%20%7Cx-x%27%7C%5Ep%29%5E%7B%5Cfrac%7B1%7D%7Bp%7D%7D%0A)

![formula](https://latex.codecogs.com/gif.latex?L_0)**范式**

即改变了多少个像素

![formula](https://latex.codecogs.com/gif.latex?L_k)**范式**

一般情况下的Minkowski距离

![formula](https://latex.codecogs.com/gif.latex?L_%5Cinfty)**范式**

改变最大的那个像素的差

#### Targeted&Nontargeted

Targeted表示欺骗了模型，使其错误地预测对抗性图像为特定标签，Nontargeted欺骗模型至其他非特定标签



这里介绍几个经典的并且较为有效的攻击

### 白箱攻击

#### Box-Constrained L-BFGS

转化为L2无约束凸优化问题

![formula](https://latex.codecogs.com/gif.latex?%0Aminimize%5Cspace%5Cspace%20c%7C%7Cx-x%27%7C%7C%5E2%2BJ_l%28x%27%2Cy%29%0A)

![formula](https://latex.codecogs.com/gif.latex?%0Ac%3E0%2Cx%27%5Cin%20%5B0%2C1%5D%5En%0A)

也可用不同的度量方式，比如![formula](https://latex.codecogs.com/gif.latex?L_%5Cinfty)。l这里指是不同的类，即同时最小化改变和对抗样本到其他类的Loss，因此也须知Target。另外这个因为可能是二阶问题的缘故采用拟牛顿法，按理GD也是可以解决的。



#### FGSM

FGSM即梯度攻击，梯度攻击目的在于让loss值增大而W，b系数不变，转化为对Loss的无约束凸优化问题。攻击表达式即

![formula](https://latex.codecogs.com/gif.latex?%0Ax%27%3Dx%2B%5Cepsilon%20sign%28%5Cnabla_xJ%28x%2Cy%29%29%0A)

让loss值增大而W，b系数不变，因此只能改变输入，我们要让输入朝Loss值变大的方向因此利用Loss对x求导，因此也需要知道Target，但这个是Nontargeted方法（[有篇论文]()上写这是targeted方法，笔者觉得从作者想表述的原理上是不对的，另外这个算法实际targeted和nontargeted都可以做）。这里取sign可能是要自己去调整系数因为毕竟扰动方向是已知的，扰动多大自己调节效果可能会好点，一般还会clip防止扰动超界。我们把FGSM的攻击式看成GD，可以看成FGSM多次更新后的结果，虽然会减慢攻击速度

![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bt%2B1%7D%3Dx_t-clip%28%5Cepsilon%20sign%28%5Cnabla_xJ%28x_t%2Cy%29%29%29%0A)

另外还有PGD算法也是多次梯度攻击

![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bt%2B1%7D%3D%5CPi_%7Bx%2BS%7D%28x_t%2B%5Cepsilon%20sign%28%5Cnabla_x%20J%28x_t%2Cy%29%29%29%0A)

注意FGSM和其变种扰动会随层数越大、线性激活函数越多而增大（毕竟BP滚雪球，另外线性函数不会让改变后的x梯度发生变化，需要保持朝向损失增大的方向）。



#### Deep Fool

先看二分类感知机，我们有扰动向量

![formula](https://latex.codecogs.com/gif.latex?%0Ar%3D-%5Cfrac%7Bf%28x_0%29%7D%7B%7C%7Cw%7C%7C_2%5E2%7Dw%0A)

其实这个公式可以理解为样本到分类边界的最短距离

![formula](https://latex.codecogs.com/gif.latex?%0A%5Cfrac%7Bf%28x_0%29%7D%7B%7C%7Cw%7C%7C_2%7D%0A)

乘上法线方向的单位向量

![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7Bw%7D%7B%7C%7Cw%7C%7C_2%7D)

，而对于多类感知机，分类正确的凸区域超平面即

![formula](https://latex.codecogs.com/gif.latex?%0AP%3D%5Ccap_c%5C%7Bx%7Cf_%7B%5Chat%7Bk%28x_0%29%7D%7D%28x%29%5Cge%20f_k%28x%29%5C%7D%0A)

到某分类决策边界的最小距离

![formula](https://latex.codecogs.com/gif.latex?%0Al%28x_0%29%3Dargmin_%7Bk%5Cneq%20%5Chat%7Bk%7D%7D%5Cfrac%7B%7Cf_k%28x_0%29-f_%7B%5Chat%7Bk%7D%7D%28x_0%29%7C%7D%7B%7C%7Cw_k-w_%7B%5Chat%7Bk%7D%7D%7C%7C_2%7D%0A)

也即最小扰动



#### C&W

我们同样需要让图片和对抗样本差距最小，而分类效果尽可能差，转为凸优化问题，即

![formula](https://latex.codecogs.com/gif.latex?%0Amin%5Cspace%20%7C%7C%5Cdelta%7C%7C_p%2Bc%20f%28x%2B%5Cdelta%29%0A)

![formula](https://latex.codecogs.com/gif.latex?%0Ax%2B%5Cdelta%20%5Cin%20%5B0%2C1%5D%5En%0A)

其中f规定小于等于0分类为其他，因此作为优化的一部分，转化扰动为

![formula](https://latex.codecogs.com/gif.latex?%0A%5Cdelta_i%3D%5Cfrac%7B1%7D%7B2%7D%28tanh%28w_i%29%2B1%29-x_n%0A)

从而定义域可以是实数域，而f函数为

![formula](https://latex.codecogs.com/gif.latex?%0Af%28x%27%29%3Dmax%28max%28%5C%7BZ%28x%27%29_i%7Ci%5Cne%20t%5C%7D-Z%28x%27%29_t%29%2C-k%29%0A)

即targeted攻击，使得攻击成t的概率增大，k变大也让分类错的概率增大



