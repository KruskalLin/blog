## Lidar Fusion

#### PLARD

![plard](img/plard.png)

LiDAR数据和RGB图像明显不是同维度的，PLARD首先将LiDAR数据投影在xy轴上转成2D图片，通过高度平均化得到一张带有高度的2D图片，之后加入线性变换网络，然后和vis融合（线性变换的时候其实就有融合了），融合方法按论文讲同Pyramid方法。即

![formula](https://latex.codecogs.com/gif.latex?f%28I%2CL%3BW%29%20%3D%20f_%7Bparsing%7D%28f_%7Bfuse%7D%20%28f_%7Bvis%7D%20%28I%3BW_%7Bvis%7D%20%29%2Cg%28L%3BW_%7Blidar%7D%20%29%29%29)

![formula](https://latex.codecogs.com/gif.latex?g%28L%3BW_%7Blidar%7D%29%20%3D%20g_%7Bfeat%7D%20%28f_%7Blidar%7D%20%28g_%7Bdata%7D%20%28L%29%3BW_%7Blidar%7D%29%29)

![formula](https://latex.codecogs.com/gif.latex?g_%7Bdata%7D%20%28L%29%7C_%7Bx%2Cy%7D%20%3D%20V_%7Bx%2Cy%7D%20%3D%20%5Cfrac%7B1%7D%7BM%7D%5Csum_%7BN_x%2CN_y%7D%5Cfrac%7BZ_%7Bx%2Cy%7D-Z_%7BN_x%2CN_y%7D%7D%7B%5Csqrt%7B%28N_x-x%29%5E2%20&plus;%20%28N_y-y%29%5E2%7D%7D)

![formula](https://latex.codecogs.com/gif.latex?g_%7Bfeat%7D%20%28f_%7Blidar%7D%20%29%20%3D%20%5Calpha%20f_%7Blidar%7D%20&plus;%20%5Cbeta)


#### LidCamNet

![lidcamnet](img/lidcamnet.png)

同样将LiDAR转换为2D数据，ReLU换成ELU，采用cross-fuse方法。即

![formula](https://latex.codecogs.com/gif.latex?L_j%5E%7Blid%7D%3DL_%7Bj-1%7D%5E%7Blid%7D&plus;a_%7Bj-1%7DL_%7Bj-1%7D%5E%7Bcam%7D)

![formula](https://latex.codecogs.com/gif.latex?L_j%5E%7Bcam%7D%3DL_%7Bj-1%7D%5E%7Bcam%7D&plus;b_%7Bj-1%7DL_%7Bj-1%7D%5E%7Blid%7D)


#### LoDNN

![lodnn](img/lodnn.png)

结合了LiDAR俯视图和FCN进行分割。主要将俯视LiDAR数据输入网络，对结果投射到road上并取得class做loss。网络结构类同Deeplab v2。



