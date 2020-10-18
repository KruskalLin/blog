### 最优化方法

就是给一个初点，找到一个收敛点列，迭代式即
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D%3Dx_k%2Ba_kd_k%0A)
其中![formula](https://latex.codecogs.com/gif.latex?a_k)为常数且大于0



#### GD

假设f(x)连续可微，并且导数在![formula](https://latex.codecogs.com/gif.latex?x_k)处不为0，我们有一阶Taylor展开
![formula](https://latex.codecogs.com/gif.latex?%0Af%28x%29%3Df%28x_k%29%2B%28x-x_k%29%5ET%5Cnabla%20f%28x_k%29%20%2B%20o%28%7C%7Cx-x_k%7C%7C%29%5Capprox%20f%28x_k%29%2B%28x-x_k%29%5ET%5Cnabla%20f%28x_k%29%0A)
这里我们需要让x随着k增大函数值减小即
![formula](https://latex.codecogs.com/gif.latex?%0Af%28x%29-f%28x_k%29%3D%28x-x_k%29%5ET%5Cnabla%20f%28x_k%29%3Da_kd_k%5ET%5Cnabla%20f%28x_k%29%3Da_kd_k%5ETg%28x_k%29%3C0%0A)
然而负数有很多，我们取最小的那个，由Cauchy-Schwarts
![formula](https://latex.codecogs.com/gif.latex?%0A%7Cd%5ET_kg%28x_k%29%7C%5Cle%20%7C%7Cd_k%7C%7C%7C%7Cg_k%7C%7C%0A)
当且仅当![formula](https://latex.codecogs.com/gif.latex?d_k%3D-g_k)时成立，因此GD推导公式为
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D%3Dx_k-a_kg_k%0A)


#### Newton

设二阶Taylor展开
![formula](https://latex.codecogs.com/gif.latex?%0Af%28x%29%5Capprox%20f%28x_k%29%2B%28x-x_k%29%5ET%5Cnabla%20f%28x_k%29%20%2B%20%5Cfrac%7B1%7D%7B2%7D%28x-x_k%29%5ET%5Cnabla%5E2%20f%28x_k%29%28x-x_k%29%0A)
同GD我们求后两项和最小值，求导有
![formula](https://latex.codecogs.com/gif.latex?%0A%5Cnabla%20f%28x_k%29%2B%5Cnabla%5E2%20f%28x_k%29%28x-x_k%29%3D0%0A)
假设Hessian矩阵![formula](https://latex.codecogs.com/gif.latex?%5Cnabla%5E2%20f%28x_k%29)这玩意儿正定，极值点为
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D%20%3D%20x_k%20-%5Cnabla%5E2%20f%28x_k%29%5E%7B-1%7D%5Cnabla%20f%28x_k%29%0A)
显然f(x)如果是二次凸函数一次迭代就到位，而对于非二次函数，牛顿法并不能保证经过有限次迭代就可以求得最优解，有一点是在极小点附近目标函数接近于二次函数，因此其收敛速度很快。对于Newton法可以加系数成为阻尼牛顿法即
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D%20%3D%20x_k%20-%5Ceta%20%5Cnabla%5E2%20f%28x_k%29%5E%7B-1%7D%5Cnabla%20f%28x_k%29%0A)



上面的两个算法都是往梯度下降最大的方向走，如果每次路线正交也可以收敛，比如CG


#### CG(共轭梯度)

对于任意n维向量![formula](https://latex.codecogs.com/gif.latex?d_i)，![formula](https://latex.codecogs.com/gif.latex?d_j)有
![formula](https://latex.codecogs.com/gif.latex?%0Ad_i%5ETGd_j%3D0%0A)
则称![formula](https://latex.codecogs.com/gif.latex?d_i)，![formula](https://latex.codecogs.com/gif.latex?d_j)关于G共轭。虽然梯度下降法的每一步都是朝着局部最优的方向前进的，但是它在不同的迭代轮数中会选择非常近似的方向，说明这个方向的误差并没通过一次更新方向和步长更新完，在这个方向上还存在误差，因此参数更新的轨迹是锯齿状。共轭梯度法的思想是，选择一个优化方向后，本次选择的步长能够将这个方向的误差更新完，在以后的优化更新过程中不再需要朝这个方向更新了。当然这玩意儿明显理想状态，现实是我们并不知道极小点在哪里，所以不可能找到准确的正交向量，否则其实可以直接求出，也就是说
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D%3Dx_k%2Ba_kd_k%0A)
我们希望![formula](https://latex.codecogs.com/gif.latex?d_k)方向上不再进行修正，因此有
![formula](https://latex.codecogs.com/gif.latex?%0Ad_k%5ETe_%7Bk%2B1%7D%3Dd_k%5ET%28e_k%2Ba_kd_k%29%3D0%0A)
但是明显![formula](https://latex.codecogs.com/gif.latex?e_k)是未知的，因而![formula](https://latex.codecogs.com/gif.latex?a_k)是求不粗来的，我们引入正交基为d的矩阵G并使得
![formula](https://latex.codecogs.com/gif.latex?%0Ad_k%5ETGe_%7Bk%2B1%7D%3D0%0A)
解得
![formula](https://latex.codecogs.com/gif.latex?%0Aa_%7Bk%7D%3D-%5Cfrac%7Bd_k%5ETGe_k%7D%7Bd_k%5ETGd_k%7D%0A)
而
![formula](https://latex.codecogs.com/gif.latex?%0AGe_k%3DG%28e_%7Bk-1%7D%2Ba_%7Bk-1%7Dd_%7Bk-1%7D%29%3DGe_%7Bk-1%7D%2BGa_%7Bk-1%7Dd_%7Bk-1%7D%0A)
 可迭代获得![formula](https://latex.codecogs.com/gif.latex?Ge_k)，另一方面由Gram-Schmidt正交化对线性无关向量![formula](https://latex.codecogs.com/gif.latex?u_1%2Cu_2...u_n)有
![formula](https://latex.codecogs.com/gif.latex?%0Ad_k%3Du_k-%5Csum%20%5Cbeta_%7Bki%7Dd_i%0A)
因此
![formula](https://latex.codecogs.com/gif.latex?%0Ad_i%5ETGd_j%3Du_i%5ETGd_j-%5Csum%20%5Cbeta_%7Bik%7Dd_k%5ETGd_j%3Du_i%5ETGd_j-%5Cbeta_%7Bij%7Dd_j%5ETGd_j%3D0%0A)
故
![formula](https://latex.codecogs.com/gif.latex?%0A%5Cbeta_%7Bij%7D%3D%5Cfrac%7Bu_i%5ETGd_j%7D%7Bd_j%5ETGd_j%7D%0A)
这玩意儿可以求。总迭代n次即收敛




#### Quasi-Newton

Newton法每次存Hessian矩阵大，且那玩意儿不一定正定，所以不一定可逆，设
![formula](https://latex.codecogs.com/gif.latex?%0Af%28x%29%5Capprox%20f%28x_k%29%2B%28x-x_k%29%5ET%5Cnabla%20f%28x_k%29%20%2B%20%5Cfrac%7B1%7D%7B2%7D%28x-x_k%29%5ET%5Cnabla%5E2%20f%28x_k%29%28x-x_k%29%0A)
求导有
![formula](https://latex.codecogs.com/gif.latex?%0A%5Cnabla%20f%28x%29%5Capprox%20%5Cnabla%20f%28x_k%29%20%2B%20%5Cnabla%5E2%20f%28x_k%29%28x-x_k%29%0A)
代入
![formula](https://latex.codecogs.com/gif.latex?%0A%5Cnabla%20f%28x_%7Bk-1%7D%29%5Capprox%20%5Cnabla%20f%28x_k%29%20%2B%20%5Cnabla%5E2%20f%28x_k%29%28x_%7Bk-1%7D-x_k%29%0A)
我们变换一下有
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D-x_k%5Capprox%20%5Cnabla%5E2f%28x_%7Bk%2B1%7D%29%5E%7B-1%7D%28%5Cnabla%20f%28x_%7Bk%2B1%7D%29-%5Cnabla%20f%28x_k%29%29%0A)
设
![formula](https://latex.codecogs.com/gif.latex?%0Ax_%7Bk%2B1%7D-x_k%3DH_%7Bk%2B1%7D%28%5Cnabla%20f%28x_%7Bk%2B1%7D%29-%5Cnabla%20f%28x_k%29%29%3Dp_k%3DH_%7Bk%2B1%7Dq_k%0A)
如果![formula](https://latex.codecogs.com/gif.latex?H_k)为n阶正定，我们构造
![formula](https://latex.codecogs.com/gif.latex?%0AH_%7Bk%2B1%7D%3DH_k%2B%5CDelta%20H_k%0A)
使得![formula](https://latex.codecogs.com/gif.latex?H_%7Bk%2B1%7D)也为正定矩阵。DFP算子
![formula](https://latex.codecogs.com/gif.latex?%0A%5CDelta%20H_k%3D%5Cfrac%7Bp_k%20p_k%5ET%7D%7Bp_k%5ETq_k%7D-%5Cfrac%7BH_kq_k%20q_k%5ETH_k%7D%7Bq_k%5ETH_kq_k%7D%0A)
这样每次就不用存每个Hessian矩阵，另外在BFGS中令![formula](https://latex.codecogs.com/gif.latex?B_%7Bk%2B1%7D%3DH_%7Bk%2B1%7D%5E%7B-1%7D)有
![formula](https://latex.codecogs.com/gif.latex?%0A%5CDelta%20B_k%20%3D%5Cfrac%7Bq_k%20q_k%5ET%7D%7Bq_k%5ETp_k%7D-%5Cfrac%7BB_kp_k%20p_k%5ETB_k%7D%7Bp_k%5ETH_kp_k%7D%0A)
再逆回来有BFGS算子
![formula](https://latex.codecogs.com/gif.latex?%0A%5CDelta%20H_k%3D%281%2B%5Cfrac%7Bq_k%5ETH_kq_k%7D%7Bp_k%5ETq_k%7D%29%5Cfrac%7Bp_k%20p_k%5ET%7D%7Bp_k%5ETq_k%7D-%5Cfrac%7Bp_k%20q_k%5ET%20H_k%2BH_kp_k%20q_k%5ET%7D%7Bp_k%5ETq_k%7D%3D%5Cfrac%7B%28p_k-H_kq_k%29p_k%5ET%2Bp_k%28p_k-H_kq_k%29%5ET%7D%7Bp_k%5ETq_k%7D-%5Cfrac%7B%28p_k-H_kq_k%29%5ETq_k%7D%7B%28p_k%5ETq_k%29%5E2%7Dp_kp_k%5ET%0A)
因此
![formula](https://latex.codecogs.com/gif.latex?%0AH_%7Bk%2B1%7D%3D%5CDelta%20H_k%2BH_k%3D%28I-%5Cfrac%7Bp_kq_k%5ET%7D%7Bp_k%5ETq_k%7D%29H_k%28I-%5Cfrac%7Bq_kp_k%5ET%7D%7Bp_k%5ETq_k%7D%29%2B%5Cfrac%7Bp_kp_k%5ET%7D%7Bp_k%5ETq_k%7D%0A)
即有迭代式，因此只需要![formula](https://latex.codecogs.com/gif.latex?H_0)和一堆的![formula](https://latex.codecogs.com/gif.latex?p_i%2Cq_i)即可算出最终结果，大幅度减少内存

