import  numpy as np

#说明python或者numpy中，尤其是构建效果时的一些不太直观的效果，做一个快速展示
print("a举例")
a=np.random.randn(5)        #产生五个高斯随机变量，储存在数组a中    a被叫做秩为1的数组，既不是行向量也不是列向量
print(a)
print(a.shape)
print(a.T)
print(np.dot(a,a.T))               #实际上得到的不是一个数字
#在编写神经网络的时候不要用形状形如上面a的形状的数据（（n,）,(,n)）
print()

#而这样
print("b举例")
b=np.random.randn(5,1)
print("b is:\n",b)      #d打印结果有两个中括号，而打印a时只有一个中括号，说明b才是真正的5x1的矩阵
print("b.T is:\n",b.T)
print("b✖b.T is:\n",np.dot(b,b.T))

#在开发一个程序时候，与其让它运行时崩溃，不如在它出现错误条件时就崩溃（返回错误）。这时候断言assert 就显得非常有用。
assert(a.shape==(5,1))

#当代码中出现了秩为1的数组的时候，可以用reshape()方法来改变数组额维度
