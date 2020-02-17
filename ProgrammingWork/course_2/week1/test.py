import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils       #第一部分，初始化
import reg_utils        #第二部分，正则化
import gc_utils         #第三部分，梯度校验

#%matplotlib inline
#设置图像的细节
plt.rcParams['figure.figsize']=(7.0,4.0)                #设置图像大小
plt.rcParams['image.interpolation']='nearest'           #这个是指定插值的方式，图像缩放之后，肯定像素要进行重新计算的，就靠这个参数来指定重新计算像素的方式
plt.rcParams['image.cmap']='gray'                       #灰度空间

#首先来看看数据集
train_X,train_Y,test_X,test_Y = init_utils.load_dataset(is_plot=True)