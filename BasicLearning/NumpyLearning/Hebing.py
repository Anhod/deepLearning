import numpy as np

#array的合并
# A=np.array([1,1,1])
# B=np.array([2,2,2])
#
# print(np.vstack((A,B)))     #把两个序列合并成了矩阵，上下合并
# print(np.hstack((A,B)))     #左右合并
#
# print(A[np.newaxis,:])      #将序列转换成一行三列的矩阵
# print(A[:,np.newaxis])      #将序列转换成三行一列的矩阵

#array的分隔
A=np.arange(12).reshape((3,4))
print(A)
print(np.split(A,3,axis=0))             #横向分割

print(np.vsplit(A, 3))          #等于 print(np.split(A, 3, axis=0))
print(np.hsplit(A, 2))          #等于 print(np.split(A, 2, axis=1))
