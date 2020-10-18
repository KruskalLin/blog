平滑性约束下的全局优化项：

![formula](https://latex.codecogs.com/gif.latex?E%28d%29%3D%5Csum_%7Bp%5Cin%20V%7DU_p%28d_p%29&plus;%5Csum_%7B%28p%2Cq%29%5Cin%20%5Cepsilon%7D%20B_%7Bpq%7D%28d_p%2Cd_q%29)

这是个整形优化问题。对于一般的二次伪布尔优化

![formula](https://latex.codecogs.com/gif.latex?E%28x%29%3D%5Csum%20E%28x_i%29%20&plus;%20%5Csum%20E%28x_i%2Cx_j%29%3D%5Csum%20a_i%20x_i%20&plus;%20%5Csum%20b_%7Bij%7Dx_ix_j%3D%5Csum_%7Bi%5Cin%20P%7Dp_ix_i-%5Csum_%7Bi%5Cin%20N%7Dp_i%5Coverline%7Bx%7D_i-%5Csum%20b_%7Bij%7Dx_i%5Coverline%7Bx%7D_j)

其中![formula](https://latex.codecogs.com/gif.latex?x%5Cin%20%5C%7B0%2C1%5C%7D%2C%5Coverline%7Bx%7D%3D1-x%2CP%3D%5C%7Bi%7Cp_i%3E0%5C%7D%2CN%3D%5C%7Bi%7Cp_i%5Cle%200%5C%7D)，令

![formula](https://latex.codecogs.com/gif.latex?%5Cepsilon%3D%5C%7B%28s%2Cx_i%3B-p_i%29%7Ci%5Cin%20N%5C%7D%5Ccup%20%5C%7B%28x_i%2Ct%3Bp_i%29%7Ci%5Cin%20P%5C%7D%20%5Ccup%20%5C%7B%28x_i%2Cx_j%3B-b_%7Bij%7D%29%5C%7D)

构建网络流，最小割即为最小能量。我们可以把整形优化转化为伪布尔优化，比如截断令

![formula](https://latex.codecogs.com/gif.latex?d%5EK%3D%20%5Cbegin%7Bequation%7D%20%5Cbegin%7Bcases%7D%200%2Cd%3CK%5C%5C%201%2Cd%5Cge%20K%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D)

那么![formula](https://latex.codecogs.com/gif.latex?d%3D%5Csum%20d%5Ei)，但这么做得到的能量式不一定符合子模性，我们利用迭代法来优化该整形，设中间迭代状态y有优化

![formula](https://latex.codecogs.com/gif.latex?E%28y%29%3D%5Csum%20E%28y_i%29&plus;%20%5Csum%20E%28y_i%2Cy_j%29)

定义一次移动为

![formula](https://latex.codecogs.com/gif.latex?y_i%3D%20%5Cbegin%7Bequation%7D%20%5Cbegin%7Bcases%7D%20%5Calpha%2Ci%3D%3Dj%5C%5C%20x_i%2Cotherwise%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D)

变为优化子模块![formula](https://latex.codecogs.com/gif.latex?E%28y_i%7Cx_%7BD-i%7D%29)，还可以有其他移动法，如![formula](https://latex.codecogs.com/gif.latex?%5Calpha)扩展

![formula](https://latex.codecogs.com/gif.latex?y_i%3D%20%5Cbegin%7Bequation%7D%20%5Cbegin%7Bcases%7D%20%5Calpha%2Cx_i%3D%3D%5Calpha%20%5C%5C%20x_i/%5Calpha%2Cotherwise%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D)

![formula](https://latex.codecogs.com/gif.latex?%5Calpha%5Cbeta)交换

![formula](https://latex.codecogs.com/gif.latex?y_i%3D%20%5Cbegin%7Bequation%7D%20%5Cbegin%7Bcases%7D%20%5Calpha/%5Cbeta%2Cx_i%3D%3D%5Cbeta/%5Calpha%20%5C%5C%20x_i%2Cotherwise%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D)

![formula](https://latex.codecogs.com/gif.latex?%5Calpha)扩展![formula](https://latex.codecogs.com/gif.latex?%5Cbeta)收缩

![formula](https://latex.codecogs.com/gif.latex?y_i%3D%20%5Cbegin%7Bequation%7D%20%5Cbegin%7Bcases%7D%20%5Calpha/%5Cbeta%2Cx_i%3D%3D%5Calpha%20%5C%5C%20%5Calpha/x_i%2Cotherwise%20%5Cend%7Bcases%7D%20%5Cend%7Bequation%7D)
