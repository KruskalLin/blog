## 数理统计基本知识回顾

### 可能重要的公式和定理


对于随机变量 ![formula](https://latex.codecogs.com/gif.latex?X%2C%20Y) 有

1. ![formula](https://latex.codecogs.com/gif.latex?Var%28X%5Cpm%20Y%29%3DVar%28X%29%2BVar%28Y%29%5Cpm%20Cov%28X%2CY%29)
2. ![formula](https://latex.codecogs.com/gif.latex?Cov%5E2%28X%2CY%29%5Cle%20Var%28X%29Var%28Y%29)
3. ![formula](https://latex.codecogs.com/gif.latex?E%28X%29%3DE%5BE%5BX%5C%7CY%5D%5D)

对于随机向量 ![formula](https://latex.codecogs.com/gif.latex?X%2CY) 有

1. ![formula](https://latex.codecogs.com/gif.latex?Cov%28X%2CY%29%3DE%28XY%5ET%29-E%28X%29E%28Y%29%5ET)
2. ![formula](https://latex.codecogs.com/gif.latex?Var%28AX%29%3DAVar%28X%29A%5ET)
3. ![formula](https://latex.codecogs.com/gif.latex?Cov%28AX%2CBY%29%3DACov%28X%2CY%29B%5ET)
4. ![formula](https://latex.codecogs.com/gif.latex?Cov%28X%2CY%29%3DCov%28Y%2CX%29%5ET)

基本大数定律与其相关（表示形式不限于）

1. ![formula](https://latex.codecogs.com/gif.latex?P%5C%7B%5C%7CX-E%5BX%5D%5C%7C%5Cge%20%5Cepsilon%20%5C%7D%20%5Cle%20%5Cfrac%7BVar%28X%29%7D%7B%5Cepsilon%5E2%7D) 
2.  ![formula](https://latex.codecogs.com/gif.latex?%5Clim%20%5Climits_%7Bn%20%5Crightarrow%20%5Cinfty%7D%20P%5C%7B%5C%7CX_n-X%5C%7C%3C%5Cepsilon%5C%7D%20%3D%201) 即 ![formula](https://latex.codecogs.com/gif.latex?X_n%20%5Cxrightarrow%7BP%7D%20X)
3. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Csum%7BX%7D%7D%7Bn%7D%20%5Cxrightarrow%7BP%7D%20%5Cmu)
4.  ![formula](https://latex.codecogs.com/gif.latex?%5Clim%20%5Climits_%7Bn%20%5Crightarrow%20%5Cinfty%7D%20F_n%28x%29%20%3D%20F%28x%29) 即 ![formula](https://latex.codecogs.com/gif.latex?F_n%28x%29%20%5Cxrightarrow%7Bw%7D%20F%28x%29) 
5. 对同测度空间有 ![formula](https://latex.codecogs.com/gif.latex?P%28%5Comega%5C%7C%5Clim%20%5Climits_%7Bn%20%5Crightarrow%20%5Cinfty%7D%20%5Cxi_n%20%28%5Comega%29%29%20%3D%20%5Cxi%20%28%5Comega%29) 即 ![formula](https://latex.codecogs.com/gif.latex?%5Cxi_n%20%5Cxrightarrow%7Ba.s.%7D%20%5Cxi) 
6. 中心极限定理常见 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Csum%7Bx_i%7D%20-%20n%5Cmu%7D%7B%5Csqrt%7Bn%7D%20%5Csigma%7D%20%5Cxrightarrow%7Bd%7D%20N%280%2C1%29)

特征函数 ![formula](https://latex.codecogs.com/gif.latex?%5Cphi_x%28t%29%3DE%5Be%5E%7BitX%7D%5D)

1. 对 ![formula](https://latex.codecogs.com/gif.latex?B%28n%2Cp%29) 特征函数为 ![formula](https://latex.codecogs.com/gif.latex?%5Cphi%28t%29%3D%28pe%5E%7Bit%7D%2Bq%29%5En)
2. 对 ![formula](https://latex.codecogs.com/gif.latex?P%28%5Clambda%29) 特征函数为 ![formula](https://latex.codecogs.com/gif.latex?%5Cphi%28t%29%3De%5E%7B%5Clambda%28e%5E%7Bit%7D-1%29%7D)
3. 对 ![formula](https://latex.codecogs.com/gif.latex?%5CGamma%28n%2C%20%5Clambda%29) 特征函数为 ![formula](https://latex.codecogs.com/gif.latex?%5Cphi%28t%29%3D%281-%5Cfrac%7Bit%7D%7B%5Clambda%7D%29%5E%7B-n%7D)
4. 对 ![formula](https://latex.codecogs.com/gif.latex?N%28%5Cmu%2C%5Csigma%5E2%29) 特征函数为 ![formula](https://latex.codecogs.com/gif.latex?%5Cphi%28t%29%3De%5E%7Bit%5Cmu-%5Cfrac%7B1%7D%7B2%7D%5Csigma%5E2t%5E2%7D)

特征函数性质

1. ![formula](https://latex.codecogs.com/gif.latex?%5C%7C%5Cphi%28t%29%5C%7C%5Cle%20%5Cphi%280%29%3D1)
2. ![formula](https://latex.codecogs.com/gif.latex?%5Cphi%28-t%29%3D%5Coverline%7B%5Cphi%28t%29%7D)
3. ![formula](https://latex.codecogs.com/gif.latex?Y%3DaX%2Bb) 则 ![formula](https://latex.codecogs.com/gif.latex?%5Cphi_Y%28t%29%3De%5E%7Bibt%7D%5Cphi_X%28at%29)
4. ![formula](https://latex.codecogs.com/gif.latex?%5Cphi_%7BX%2BY%7D%28t%29%3D%5Cphi_X%28t%29%5Cphi_Y%28t%29)

矩阵相关 (numerator layout)

1. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20Ax%7D%7B%5Cpartial%20x%7D%3DA)
2. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20x%5ETA%7D%7Bx%7D%3DA%5ET)
3. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20x%5ETx%7D%7B%5Cpartial%20x%7D%3D2x)
4. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20x%5ETAx%7D%7B%5Cpartial%20x%7D%3Dx%5ET%28A%2BA%5ET%29)
5. ![formula](https://latex.codecogs.com/gif.latex?tr%28AB%29%3DBA)
6. ![formula](https://latex.codecogs.com/gif.latex?tr%28ABC%29%3Dtr%28BCA%29%3Dtr%28CAB%29)
7. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20tr%28AB%29%7D%7B%5Cpartial%20A%7D%3DB%5ET)
8. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%20ABA%5ETC%7D%7B%5Cpartial%20A%7D%3DCAB%2BC%5ETAB%5ET)

### 可能重要的分布与其性质

#### 离散型

1. 二项分布 ![formula](https://latex.codecogs.com/gif.latex?B%28n%2Cp%29) ，其中 ![formula](https://latex.codecogs.com/gif.latex?f%28k%29%3DC_%7Bn%7D%5Ekp%5E%7Bk%7D%281-p%29%5E%7Bn-k%7D), ![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3Dnp), ![formula](https://latex.codecogs.com/gif.latex?D%28%5Cxi%29%3Dnp%281-p%29)，可加，![formula](https://latex.codecogs.com/gif.latex?n%3D1) 时为 ![formula](https://latex.codecogs.com/gif.latex?0-1)分布，渐进分布为泊松分布
2. 泊松分布 ![formula](https://latex.codecogs.com/gif.latex?P%28%5Clambda%29) ，其中 ![formula](https://latex.codecogs.com/gif.latex?f%28k%29%3D%5Cfrac%7B%5Clambda%5Ek%7D%7Bk%21%7De%5E%7B-%5Clambda%7D) ，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Clambda)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%5E2%29%3D%5Clambda%5E2%2B%5Clambda)， ![formula](https://latex.codecogs.com/gif.latex?D%28%5Cxi%29%3D%5Clambda)，可加
3. 巴斯卡分布 ![formula](https://latex.codecogs.com/gif.latex?N_b%28r%2Cp%29) ，其中 ![formula](https://latex.codecogs.com/gif.latex?f%28k%29%3DC_%7Bk-1%7D%5E%7Br-1%7Dp%5Er%281-p%29%5E%7Bk-r%7D)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Cfrac%7Br%7D%7Bp%7D)，![formula](https://latex.codecogs.com/gif.latex?D%28%5Cxi%29%3D%5Cfrac%7Brp%281-p%29%7D%7Bp%5E2%7D)，![formula](https://latex.codecogs.com/gif.latex?r%3D1) 时为几何分布
4. 超几何分布 ![formula](https://latex.codecogs.com/gif.latex?H%28n%2CM%2CN%29)，其中 ![formula](https://latex.codecogs.com/gif.latex?f%28k%29%3D%5Cfrac%7BC_%7BM%7D%5E%7Bk%7DC_%7BN-M%7D%5E%7Bn-k%7D%7D%7BC_%7BN%7D%5E%7Bn%7D%7D)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Cfrac%7BnM%7D%7BN%7D)

#### 连续型

1. 均匀分布 ![formula](https://latex.codecogs.com/gif.latex?U%5Ba%2Cb%5D) 其中 ![formula](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B1%7D%7Bb-a%7D%28a%5Cle%20x%20%5Cle%20b%29) ，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Cfrac%7Ba%2Bb%7D%7B2%7D)，![formula](https://latex.codecogs.com/gif.latex?D%28%5Cxi%29%3D%5Cfrac%7B%28b-a%29%5E2%7D%7B12%7D)
2. 正态分布 ![formula](https://latex.codecogs.com/gif.latex?N%28%5Cmu%2C%5Csigma%5E2%29) 其中 ![formula](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B1%7D%7B%5Csqrt%7B2%5Cpi%7D%5Csigma%7De%5E%7B-%5Cfrac%7B%28x-%5Cmu%29%5E2%7D%7B2%5Csigma%5E2%7D%7D)
3. ![formula](https://latex.codecogs.com/gif.latex?%5CGamma) 分布 ![formula](https://latex.codecogs.com/gif.latex?%5CGamma%28%5Calpha%2C%5Clambda%29) 其中 ![formula](https://latex.codecogs.com/gif.latex?%5CGamma%28%5Calpha%29%3D%5Cint%5E%7B%2B%5Cinfty%7D_%7B0%7Dx%5E%7B%5Calpha-1%7De%5E%7B-x%7Ddx) ，![formula](https://latex.codecogs.com/gif.latex?%5CGamma%281%29%3D1)，![formula](https://latex.codecogs.com/gif.latex?%5CGamma%28%5Cfrac%7B1%7D%7B2%7D%29%3D%5Csqrt%7B%5Cpi%7D)，![formula](https://latex.codecogs.com/gif.latex?%5CGamma%28n%29%3D%28n-1%29%21%28n%5Cin%20N%5E%2B%29)，![formula](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B%5Clambda%5E%5Calpha%7D%7B%5CGamma%28%5Calpha%29%7Dx%5E%7B%5Calpha-1%7De%5E%7B-%5Clambda%20x%7D%28x%5Cge0%29)，![formula](https://latex.codecogs.com/gif.latex?%5CGamma%281%2C%5Clambda%29%3DExp%28%5Clambda%29)，![formula](https://latex.codecogs.com/gif.latex?%5CGamma%28%5Cfrac%20n2%2C%5Cfrac%2012%29%3D%5Cchi%5E2%28n%29)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Cfrac%20%5Calpha%5Clambda)，![formula](https://latex.codecogs.com/gif.latex?D%28%5Cxi%29%3D%5Cfrac%7B%5Calpha%7D%7B%5Clambda%5E2%7D)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%5Ek%29%3D%5Cfrac%7B%5CGamma%28%5Calpha%2Bk%29%7D%7B%5CGamma%28%5Calpha%29%5Clambda%5Ek%7D)，可加
4. 逆 ![formula](https://latex.codecogs.com/gif.latex?%5CGamma) 分布，其中 ![formula](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B%5Clambda%5E%5Calpha%7D%7B%5CGamma%28%5Calpha%29x%5E%7B%5Calpha%2B1%7D%7De%5E%7B-%5Cfrac%20%5Clambda%20x%7D)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Cfrac%7B%5Clambda%7D%7B%5Calpha-1%7D)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%5Ek%29%3D%5Cfrac%7B%5Clambda%5Ek%20%5CGamma%28%5Calpha-k%29%7D%7B%5CGamma%28%5Calpha%29%7D)
5. ![formula](https://latex.codecogs.com/gif.latex?%5Cbeta) 分布，其中 ![formula](https://latex.codecogs.com/gif.latex?B%28a%2Cb%29%3D%5Cint%5E1_0x%5E%7Ba-1%7D%281-x%29%5E%7Bb-1%7D%3D%5Cfrac%7B%5CGamma%28%5Calpha%29%5CGamma%28b%29%7D%7B%5CGamma%28a%2Bb%29%7D)，![formula](https://latex.codecogs.com/gif.latex?B%28a%2Cb%29%3DB%28b%2Ca%29)，![formula](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7Bx%5E%7Ba-1%7D%281-x%29%5E%7Bb-1%7D%7D%7BB%28a%2Cb%29%7D%280%5Cle%20x%20%5Cle%201%29)，![formula](https://latex.codecogs.com/gif.latex?%5Cbeta%281%2C1%29%3DU%5B0%2C1%5D)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cxi%29%3D%5Cfrac%7Ba%7D%7Ba%2Bb%7D)，![formula](https://latex.codecogs.com/gif.latex?D%28%5Cxi%29%3D%5Cfrac%7Bab%7D%7B%28a%2Bb%29%5E2%28a%2Bb%2B1%29%7D)
6. ![formula](https://latex.codecogs.com/gif.latex?%5Cchi%5E2) 分布，实际上 ![formula](https://latex.codecogs.com/gif.latex?%5Cchi%5E2%28n%29%5Csim%5CGamma%28%5Cfrac%20n2%2C%5Cfrac%2012%29)
7. ![formula](https://latex.codecogs.com/gif.latex?F) 分布，对于 ![formula](https://latex.codecogs.com/gif.latex?F%28m%2Cn%29)，有 ![formula](https://latex.codecogs.com/gif.latex?F%3D%5Cfrac%7BnX%7D%7BmY%7D) 其中 ![formula](https://latex.codecogs.com/gif.latex?X%5Csim%5Cchi%5E2%28m%29)，![formula](https://latex.codecogs.com/gif.latex?Y%5Csim%5Cchi%5E2%28n%29)，另外 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%201F%20%5Csim%20F%28n%2Cm%29)
8. ![formula](https://latex.codecogs.com/gif.latex?t) 分布，对于 ![formula](https://latex.codecogs.com/gif.latex?t%28n%29) 有 ![formula](https://latex.codecogs.com/gif.latex?T%3D%5Cfrac%7BX%7D%7B%5Csqrt%7B%5Cfrac%20Yn%7D%7D)，其中 ![formula](https://latex.codecogs.com/gif.latex?X%5Csim%20N%280%2C1%29)，![formula](https://latex.codecogs.com/gif.latex?Y%5Csim%5Cchi%5E2%28n%29)，![formula](https://latex.codecogs.com/gif.latex?X%5E2%20%5Csim%20F%281%2Cn%29)，此外 ![formula](https://latex.codecogs.com/gif.latex?T_n%20%5Cxrightarrow%7Bp%7D%20N%280%2C1%29)
9. 多维正态分布 ![formula](https://latex.codecogs.com/gif.latex?f%28%5Ceta%29%3D%5Cfrac%7B1%7D%7B%282%5Cpi%29%5E%7B%5Cfrac%20n2%7D%5C%7C%5CSigma%5C%7C%5E%7B%5Cfrac%2012%7D%7De%5E%7B-%5Cfrac%2012%20%28%5Ceta-%5Ctheta%29%5ET%5CSigma%5E%7B-1%7D%28%5Ceta-%5Ctheta%29%7D)，有性质
   1. 对正交矩阵 ![formula](https://latex.codecogs.com/gif.latex?T)，正态分布 ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%20%5Csim%20N%28%5Ctheta%2C%5Csigma%5E2I_n%29)，有 ![formula](https://latex.codecogs.com/gif.latex?T%27%28%5Cfrac%7B%5Ceta-%5Ctheta%7D%7B%5Csigma%7D%29%5Csim%20N%280%2CI_n%29)
   2. 对对称幂等矩阵 ![formula](https://latex.codecogs.com/gif.latex?A)，正态分布 ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%20%5Csim%20N%28%5Ctheta%2C%5Csigma%5E2I_n%29)，![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%28%5Ceta-%5Ctheta%29%27A%28%5Ceta-%5Ctheta%29%7D%7B%5Csigma%5E2%7D%5Csim%20%5Cchi%5E2%20%28tr%28A%29%29)
   3. ![formula](https://latex.codecogs.com/gif.latex?%5Cxi%3DA%5Ceta%2B%5Calpha%5Csim%20N%28A%5Ctheta%2B%5Calpha%2CA%5CSigma%20A%27%29)
   4. 对称矩阵 ![formula](https://latex.codecogs.com/gif.latex?A)，正态分布 ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%20%5Csim%20N%28%5Ctheta%2C%5Csigma%5E2I_n%29)，若 ![formula](https://latex.codecogs.com/gif.latex?BA%3D0) 则 ![formula](https://latex.codecogs.com/gif.latex?B%5Ceta) 和 ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%27A%5Ceta) 相互独立，若 ![formula](https://latex.codecogs.com/gif.latex?B) 亦为同阶对称矩阵，则 ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%27A%5Ceta) 和 ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%27B%5Ceta) 相互独立
   5. ![formula](https://latex.codecogs.com/gif.latex?%5Ceta%20%5Csim%20N%28%5Ctheta%2C%5CSigma%29) 则 ![formula](https://latex.codecogs.com/gif.latex?%28%5Ceta-%5Ctheta%29%27%5CSigma%5E%7B-1%7D%28%5Ceta-%5Ctheta%29%5Csim%20%5Cchi%5E2%28n%29)


对于连续型分布，若存在 ![formula](https://latex.codecogs.com/gif.latex?g%28x%29)，有![formula](https://latex.codecogs.com/gif.latex?x%3Dg%5E%7B-1%7D%28y%29)，有 ![formula](https://latex.codecogs.com/gif.latex?f_Y%28y%29%3Df_X%28g%5E%7B-1%7D%28y%29%29%5C%7Cg%5E%7B-1%7D%28y%29%27%5C%7C) ，以此容易证明

1. 对于 ![formula](https://latex.codecogs.com/gif.latex?X%5Csim%5CGamma%28%5Calpha%2C%5Clambda%29)， ![formula](https://latex.codecogs.com/gif.latex?2%5Calpha%20%5Cin%20N%5E%2B)，有 ![formula](https://latex.codecogs.com/gif.latex?2%5Clambda%20X%5Csim%20%5Cchi%5E2%282%5Calpha%29)
2. 对于 ![formula](https://latex.codecogs.com/gif.latex?X%5Csim%20U%5B0%2C1%5D) 有 ![formula](https://latex.codecogs.com/gif.latex?-lnX%20%5Csim%20Exp%281%29)
3. 对于 ![formula](https://latex.codecogs.com/gif.latex?X%5Csim%20U%5B0%2C1%5D) 有 ![formula](https://latex.codecogs.com/gif.latex?-2lnX%20%5Csim%20%5Cchi%5E2%282n%29)

统计分布常用性质

1. ![formula](https://latex.codecogs.com/gif.latex?%5Coverline%7B%5Cxi%7D%20%5Csim%20N%28%5Ctheta%2C%5Cfrac%7B%5Csigma%5E2%7D%7Bn%7D%29)
2. ![formula](https://latex.codecogs.com/gif.latex?%5Coverline%7B%5Cxi%7D) 与 ![formula](https://latex.codecogs.com/gif.latex?S%5E2) 相互独立
3. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BnS%5E2%7D%7B%5Csigma%5E2%7D%5Csim%20%5Cchi%5E2%28n-1%29)
4. ![formula](https://latex.codecogs.com/gif.latex?T%3D%5Cfrac%7B%5Coverline%28%5Cxi%29-%5Ctheta%7D%7BS/%5Csqrt%7Bn-1%7D%7D%5Csim%20t%28n-1%29)

### 可能重要的统计量性质

#### 顺序统计量

1. ![formula](https://latex.codecogs.com/gif.latex?f_%7B%28i%29%7D%3D%5Cfrac%7Bn%21%7D%7B%28i-1%29%21%28n-i%29%21%7Df%28x%29F%28x%29%5E%7Bi-1%7D%281-F%28X%29%29%5E%7Bn-i%7D)
2. ![formula](https://latex.codecogs.com/gif.latex?F_%7B%28n%29%7D%3DF%28x%29%5En)
3. ![formula](https://latex.codecogs.com/gif.latex?F_%7B%281%29%7D%3D1-%281-F%28x%29%29%5En)
4. 若 ![formula](https://latex.codecogs.com/gif.latex?%5Cxi%20%5Csim%20U%5B0%2C1%5D) 则 ![formula](https://latex.codecogs.com/gif.latex?%5Cxi%28i%29%5Csim%20%5Cbeta%28i%2Cn-i%2B1%29)

#### 经验分布

1. ![formula](https://latex.codecogs.com/gif.latex?nF_n%28x%29%5Csim%20B%28n%2CF%28x%29%29)
2. ![formula](https://latex.codecogs.com/gif.latex?F_n%28x%29%20%5Cxrightarrow%7Bp%7D%20F%28x%29)
3. 渐进有 ![formula](https://latex.codecogs.com/gif.latex?F_n%28x%29%20%5Csim%20N%28F%28x%29%2C%5Cfrac%7BF%28x%29%281-F%28x%29%29%7D%7Bn%7D%29) 

## 估计理论

我们不可能得到所有的数据，因此要用一些手段来推测真实分布

对于参数点估计，我们可以做到

1. 减少候选估计项，即取UMVUE
   1. 观察有效性时一般会用到 C-R 不等式，即对 ![formula](https://latex.codecogs.com/gif.latex?g%28%5Ctheta%29) 和其无偏估计量 ![formula](https://latex.codecogs.com/gif.latex?T)
      1. ![formula](https://latex.codecogs.com/gif.latex?D_%5Ctheta%20%28T%29%5Cge%20%5Cfrac%7B%5Bg%27%28%5Ctheta%29%5D%5E2%7D%7BnI%28%5Ctheta%29%7D) 其中 ![formula](https://latex.codecogs.com/gif.latex?I%28%5Ctheta%29%3DE_%5Ctheta%5B%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20%5Ctheta%7Dlnf%28%5Cxi%3B%5Ctheta%29%5D%5E2%3D-E_%5Ctheta%5B%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%5Ctheta%5E2%7Dlnf%28%5Cxi%3B%5Ctheta%29%5D)，如果 ![formula](https://latex.codecogs.com/gif.latex?g%28%5Ctheta%29%3D%5Ctheta)，则 ![formula](https://latex.codecogs.com/gif.latex?D_%5Ctheta%20%28T%29%5Cge%20%5Cfrac%7B1%7D%7BnI%28%5Ctheta%29%7D)
         2. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%5Ctheta%7D%5Bln%20%5CPi%20f%28%5Cxi%3B%5Ctheta%29%5D%3DC%28%5Ctheta%29%5BT-g%28%5Ctheta%29%5D)，这也是有有效估计量的充要条件
   2. 观察一致性时一般用 Chebyshev 或者大数定律，对于无偏估计一般证明其均方一致比较容易
2. 弱化最优定义，极大似然估计
3. 贝叶斯估计，引入先验分布选取最优估计，或者最大化后验分布概率，均方损失下贝叶斯估计为后验期望，绝对值损失下为后验中位数（这损失有点像1，2范数），贝叶斯估计基本就是对照密度函数，写出分布

对于参数区间估计，一般先选取枢轴量 U（这里是分布已知的估计量或者其变化，如果没有给出枢轴量，可以尝试通过构造函数导出新的统计量，或者求极大似然估计等），然后取到区间使得 ![formula](https://latex.codecogs.com/gif.latex?P%28C_1%3CU%3CC_2%29%3D%5Calpha)

具体做法书上有表格，略

## 假设检验

第一类错误：![formula](https://latex.codecogs.com/gif.latex?%5Calpha%3DP%28reject%5C%20H_0%5C%7CH_0%5C%20is%5C%20true%29) ，第二类错误：![formula](https://latex.codecogs.com/gif.latex?%5Cbeta%3DP%28accept%5C%20H_0%5C%7CH_0%5C%20is%5C%20false%29) ，一般来说 ![formula](https://latex.codecogs.com/gif.latex?%5Calpha) 变动会引起 ![formula](https://latex.codecogs.com/gif.latex?%5Cbeta) 的变动，因而一般情况下固定一个值，另外一般情况下题目的两个假设是互补的，拒绝第一个假设意味着接受第二个假设。

### 参数假设检验

一般先构造统计量，然后预估统计量的范围，通过统计量的分布和显著性水平确定范围

具体做法书上有表格，略

### 非参假设

1. ![formula](https://latex.codecogs.com/gif.latex?%5Cchi%5E2) 检验法，构造 m 个连续区间，设统计量 ![formula](https://latex.codecogs.com/gif.latex?%5Cchi%5E2_n%3D%5Csum%7B%5Cfrac%7Bv%5E2_i%7D%7Bnp_i%7D%7D-n) 其中 ![formula](https://latex.codecogs.com/gif.latex?v) 为频数，![formula](https://latex.codecogs.com/gif.latex?p_i) 为假设分布的在某区间下的概率，拒绝域 ![formula](https://latex.codecogs.com/gif.latex?%20%5Cchi_0%20%3D%20%5B%5Cchi%5E2_n%20%3E%20%5Cchi%5E2_%7B%201%20-%20%5Calpha%20%7D%20%2A%20%28m-1-l%29%5D%20) 其中 ![formula](https://latex.codecogs.com/gif.latex?l) 为未知的参数个数，未知参数可以通过极大似然估计等求得。书中的独立性检验是他的一个应用
2. 单总体双总体 K.S. 检验，即求所有区间中分布之间差最大的值，太大则可能不是同一个分布
3. 秩和检验，从 1 开始排序，计算第一个总体的 rank，总体的 rank 不能太大也不能太小，因为两个相近的分布，其变量一般是交错的
4. 游程检验，思想类似秩和检验，排序后计算有多少个连续的 0 或者 1 序列，序列数不能太小

## 方差分析

### 单因素

即检验多个正态水平的均值是否一致，记 ![formula](https://latex.codecogs.com/gif.latex?Q_A%3D%5Csum%20n_i%28%5Coverline%7B%5Ceta_i%7D-%5Coverline%7B%5Ceta%7D%29%5E2) 即水平间差距，![formula](https://latex.codecogs.com/gif.latex?Q_e%3D%5Csum%20n_i%20S%5E2_i) 即水平内误差，有
![formula](https://latex.codecogs.com/gif.latex?%0AF%3D%5Cfrac%7BQ_A/%28r-1%29%7D%7BQ_e/%28n-r%29%7D%20%5Csim%20F%28r-1%2Cn-r%29%0A)
如果F太大则证明水平间差距大，因而拒绝原假设。故有拒绝域 ![formula](https://latex.codecogs.com/gif.latex?%5Cchi_0%3D%5BF%3EF_%7B1-%5Calpha%7D%28r-1%2Cn-r%29%5D)

## 一元线性回归

一般形式 ![formula](https://latex.codecogs.com/gif.latex?y%3D%5Cbeta_0%2B%5Cbeta_1x%2B%5Cepsilon) ，其中 ![formula](https://latex.codecogs.com/gif.latex?%5Cepsilon%20%5Csim%20N%280%2C%20%5Csigma%5E2%29)， 记 ![formula](https://latex.codecogs.com/gif.latex?l_%7Bxy%7D%3D%5Csum%20%28x-%5Coverline%7Bx%7D%29%28y-%5Coverline%7By%7D%29) ，![formula](https://latex.codecogs.com/gif.latex?l_%7Bxx%7D%3D%5Csum%20%28x-%5Coverline%7Bx%7D%29%5E2)，![formula](https://latex.codecogs.com/gif.latex?l_%7Byy%7D%3D%5Csum%28y-%5Coverline%7By%7D%29%5E2)
![formula](https://latex.codecogs.com/gif.latex?%0AL%3DX%5ETX%3D%5Cbegin%7Bbmatrix%7D%0An%20%26%20n%5Coverline%7Bx%7D%5C%5C%0An%5Coverline%7Bx%7D%20%26%20%5Csum%20x_i%5E2%0A%5Cend%7Bbmatrix%7D%0A)
求逆有

![formula](https://latex.codecogs.com/gif.latex?%0AL%5E%7B-1%7D%3D%5Cfrac%7B1%7D%7Bnl_%7Bxx%7D%7D%0A%5Cbegin%7Bbmatrix%7D%0A%5Csum%20x_i%5E2%20%26%20-n%5Coverline%7Bx%7D%5C%5C%0A-n%5Coverline%7Bx%7D%20%26%20n%0A%5Cend%7Bbmatrix%7D%0A)
利用 ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta%7D%3DL%5E%7B-1%7DX%27Y)，那么有 ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D%20%3D%20%5Cfrac%7Bl_%7Bxy%7D%7D%7Bl_%7Bxx%7D%7D)，![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_0%7D%3D%5Coverline%7By%7D-%5Chat%7B%5Cbeta_1%7D%5Coverline%7Bx%7D)，![formula](https://latex.codecogs.com/gif.latex?U_R%3D%5Csum%28%5Chat%7By_i%7D-%5Coverline%7By%7D%29%3D%5Chat%7B%5Cbeta_1%7Dl_%7Bxy%7D)，![formula](https://latex.codecogs.com/gif.latex?Q_e%3De%27e%3Dl_%7Byy%7D-U_R)，其他性质有

1. ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D%20%5Csim%20N%28%5Cbeta_1%2C%5Cfrac%7B%5Csigma%5E2%7D%7Bl_%7Bxx%7D%7D%29)
2. ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D) 与 ![formula](https://latex.codecogs.com/gif.latex?%5Coverline%7By%7D) 独立
3. ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_0%7D%20%5Csim%20N%28%5Cbeta_0%2C%28%5Cfrac%7B1%7D%7Bn%7D%2B%5Cfrac%7B%5Coverline%28x%29%7D%7Bl_%7Bxx%7D%7D%29%5Csigma%5E2%29)
4. ![formula](https://latex.codecogs.com/gif.latex?E%28Q_e%29%3D%28n-2%29%5Csigma%5E2)，记 ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Csigma%5E2_e%7D%3DQ_e/%28n-2%29) 有 ![formula](https://latex.codecogs.com/gif.latex?E%28%5Chat%7B%5Csigma%5E2_e%7D%29%3D%5Csigma%5E2)，此外 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BQ_e%7D%7B%5Csigma%5E2%7D%5Csim%20%5Cchi%5E2%28n-2%29)
5. ![formula](https://latex.codecogs.com/gif.latex?Q_e) 与 ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_0%7D)，![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D)，![formula](https://latex.codecogs.com/gif.latex?%5Chat%7By_i%7D) 均独立
6. 可以利用 ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_1%7D)，![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta_0%7D) 的分布和性质 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BQ_e%7D%7B%5Csigma%5E2%7D%5Csim%20%5Cchi%5E2%28n-2%29) 构造 T 或 F 统计量
7. ![formula](https://latex.codecogs.com/gif.latex?cov%28%5Chat%7B%5Cbeta_0%7D%2C%5Chat%7B%5Cbeta_1%7D%29%3D-%5Cfrac%7B%5Coverline%7Bx%7D%7D%7Bl_%7Bxx%7D%7D%5Csigma%5E2)，因而 ![formula](https://latex.codecogs.com/gif.latex?%5Coverline%7Bx%7D%3D0) 时两者相互独立
8. ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta%7D) 为 MLE，BLVE，提及 BLVE 是因为 ![formula](https://latex.codecogs.com/gif.latex?%5Chat%7B%5Cbeta%7D) 可以看作 Y 的 函数

预测控制

1. 点预测，直接代入
2. 区间预测 ![formula](https://latex.codecogs.com/gif.latex?y-%5Chat%7By%7D%20%5Csim%20N%280%2C%5Csigma%5E2%281%2B%5Cfrac%201n%20%2B%20%5Cfrac%7B%28x-%5Coverline%7Bx%7D%29%5E2%7D%7Bl_%7Bxx%7D%7D%29%29)，利用 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BQ_e%7D%7B%5Csigma%5E2%7D%5Csim%20%5Cchi%5E2%28n-2%29) 有 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7By-%5Chat%7By%7D%7D%7B%5Chat%7B%5Csigma%5E2_e%7D%5Csqrt%7B1%2B%5Cfrac%201n%20%2B%20%5Cfrac%7B%28x-%5Coverline%7Bx%7D%29%5E2%7D%7Bl_%7Bxx%7D%7D%7D%7D)
3. 控制，反解上式，一般会取 ![formula](https://latex.codecogs.com/gif.latex?%5Csqrt%7B1%2B%5Cfrac%201n%20%2B%20%5Cfrac%7B%28x-%5Coverline%7Bx%7D%29%5E2%7D%7Bl_%7Bxx%7D%7D%7D%3D1) 与 ![formula](https://latex.codecogs.com/gif.latex?t_%7B1-%5Cfrac%20%5Calpha%202%7D%20%28n-2%29%3D%5Cmu_%7B1-%5Cfrac%20%5Calpha%202%7D) 即控制范围 ![formula](https://latex.codecogs.com/gif.latex?%5B%5Chat%7B%5Cbeta_0%7D%2B%5Chat%7B%5Cbeta_1%7Dx-%5Chat%7B%5Csigma%5E2_e%7D%5Cmu_%7B1-%5Cfrac%20%5Calpha%202%7D%2C%5Chat%7B%5Cbeta_0%7D%2B%5Chat%7B%5Cbeta_1%7Dx%2B%5Chat%7B%5Csigma%5E2_e%7D%5Cmu_%7B1-%5Cfrac%20%5Calpha%202%7D%5D)



多元情况下有一般情况 ![formula](https://latex.codecogs.com/gif.latex?Y%3DX%5Cbeta%2B%5Cepsilon)，![formula](https://latex.codecogs.com/gif.latex?E%28%5Cepsilon%29%3D0) 且 ![formula](https://latex.codecogs.com/gif.latex?cov%28%5Cepsilon%2C%5Cepsilon%29%3D%5Csigma%5E2I_n)，有重要性质

1. ![formula](https://latex.codecogs.com/gif.latex?cov%28%5Chat%7B%5Cbeta%7D%2C%5Chat%7B%5Cbeta%7D%29%3D%5Csigma%5E2L%5E%7B-1%7D)
2. ![formula](https://latex.codecogs.com/gif.latex?cov%28%5Chat%7B%5Cbeta%7D%2Ce%29%3D0)，![formula](https://latex.codecogs.com/gif.latex?cov%28%5Chat%7B%5Cbeta%7D%2CQ_e%29%3D0)
3. ![formula](https://latex.codecogs.com/gif.latex?%7B%5Chat%7B%5Cbeta%7D%7D) 为 BLVE
4. ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BQ_e%7D%7B%5Csigma%5E2%7D%5Csim%5Cchi%5E2%28n-k-1%29)
5. ![formula](https://latex.codecogs.com/gif.latex?%5Cbeta%20%5Csim%20N_%7Bk%2B1%7D%28%5Cbeta%2C%5Csigma%5E2L%5E%7B-1%7D%29) 可以通过此式构造很多形式的T或F统计量
6. 记 ![formula](https://latex.codecogs.com/gif.latex?U%3D%5Chat%7BY%7D%27%5Chat%7BY%7D)，则 U 和 ![formula](https://latex.codecogs.com/gif.latex?Q_e) 独立且 ![formula](https://latex.codecogs.com/gif.latex?%5Cbeta%3D0) 有 ![formula](https://latex.codecogs.com/gif.latex?%5Cfrac%7BU%7D%7B%5Csigma%5E2%7D%5Csim%5Cchi%5E2%28k%2B1%29)

如果有约束 ![formula](https://latex.codecogs.com/gif.latex?H%5Cbeta%3Dd)，其中 H 是 ![formula](https://latex.codecogs.com/gif.latex?q%5Ctimes%28k%2B1%29) 的矩阵有

![formula](https://latex.codecogs.com/gif.latex?%0A%5Cfrac%7B%28Q_%7Be_%7BH%7D%7D-Q_e%29/q%7D%7BQ_e/%28n-k-1%29%7D%3D%5Cfrac%7Bn-k-1%7D%7Bq%7D%5Cfrac%7B%28H%5Chat%7B%5Cbeta%7D-d%29%27%28HL%5E%7B-1%7DH%27%29%5E%7B-1%7D%28H%5Chat%7B%5Cbeta%7D-d%29%7D%7BQ_e%7D%5Csim%20F%28q%2Cn-k-1%29%0A)
