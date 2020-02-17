import numpy as np

a=np.array([10,20,30,40])
b=np.arange(4)

print(a,b)
print(a-b)
print(a*b)
print(b<3)                      #对每一个值进行判断

#对于矩阵
a=np.array([[1,2],[3,4]])
b=np.arange(2,6).reshape((2,2))

c=a*b
c_dot=np.dot(a,b)    #equals  c_dot=a.dot(b)
print(c)                                          #逐个元素相乘
print(c_dot)                                      #矩阵的乘法

#生成随机矩阵
a=np.random.random((2,3))                         #参数是shape
print(a)
print(np.sum(a,axis=1))                           #按行
print(np.min(a,axis=0))                           #按列
print(np.max(a,axis=1))