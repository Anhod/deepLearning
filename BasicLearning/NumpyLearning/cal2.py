import numpy as np

A=np.array([[2,3,4,5],
            [44,33,5,22],
            [2,7,0,5]])
print("argmin:\n",np.argmin(A))          #矩阵中最小值的索引
print("mean:\n",np.mean(A))             #平均值，或者（np.average(A)）   或者(A.mean()),可以指定行列
print("median:\n",np.median(A))         #中位数

print(A)
print("cunsum:\n",np.cumsum(A))         #第几个元素就是A中的前几个元素累加
print("diff:\n",np.diff(A))             #累差
print("nonzero:\n",np.nonzero(A))       #输出两个array，其中，第一个array是行数，第二个是列数，两个array中的第一个元素组合起来，则是一个非零元素在矩阵中的位置
print("sort:\n",np.sort(A))             #逐行排序
print("transpose:\n",A.T)
print("clip；\n",np.clip(A,5,22))          #矩阵中小于5的数都变成5，大于22的数都变成22