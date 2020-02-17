import  numpy as np

#course1
array=np.array([[1,2,3],[2,3,4]])                   #将列表转换为矩阵
print(array)
print("number of dim:",array.ndim)                  #维数
print("shape:",array.shape)                         #形状，也就是行数和列数
print("size：",array.size)                          #元素个数

#course2
a=np.array([1,2,3,4],dtype=np.int)                  #int32  int64  float32  float64等等数据类型
print("dtype:",a.dtype)

b=np.zeros((4,3))                                   #括号里面是维度
print("零矩阵：\n",b)

c=np.ones((3,4))
print("全部为1的矩阵：\n",c)

d=np.arange(1,10).reshape((3,3))                    #默认的shape是一行多列，运用reshape（）可以改变形状
print(d)

e=np.linspace(1,10,20)                              #生成从1-10的20段的线段，也可以运用reshape函数
print(e)


