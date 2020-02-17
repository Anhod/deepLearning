import numpy as np

A=np.array([[56.0,0.0,4.4,68.0],[1.2,104,52.0,8.0],[1.8,135.0,99.0,0.9]])
print("The matrix A is:\n",A)
print()

cal=A.sum(axis=0)                       #axis=0垂直方向上对矩阵求和
print("The cal is:\n",cal)
print()

percentage=100*A/cal.reshape(1,4)       #除以原矩阵的每一项，得到一个比例
print("The percentage is:\n",percentage)


'''
补充：
    再reshape()函数中，如果有-1参数，则是想根据另一个参数的值算出本该在-1位置上的值
    例如： z = np.array([[1, 2, 3, 4],[5, 6, 7, 8],[9, 10, 11, 12],[13, 14, 15, 16]])
     print(z.reshape(-1,1))
     我们不知道z的shape属性是多少， 但是想让z变成只有一列，行数不知道多少，
    通过`z.reshape(-1,1)`，Numpy自动计算出有16行，新的数组shape属性为(16, 1)，与原来的(4, 4)配套。
'''