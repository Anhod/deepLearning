import numpy as np

A=np.arange(3,15)
print(A)
print(A[3])

A=np.arange(3,15).reshape(3,4)
print(A[2])                 #索引的是行数
print(A[1][1])              #第一行第一列
print(A[1,1])               #第一行第一列
print(A[2,:])               #第二行的所有数
print(A[:,1])               #第一列的所有数
print(A[1,1:2])             #第一行的第一到第二个元素（不包括索引2）

for row in A:
    print(row)

for column in A.T:
    print(column)

print(A.flatten())              #返回array
for item in A.flat:
    print(item)
