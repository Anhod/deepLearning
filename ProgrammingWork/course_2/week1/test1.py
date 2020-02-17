import numpy as np

a = np.array([[1,2,3],
     [4,5,6]])
a = a.reshape((-1,1))

b = np.array([[7,8,9],
              [10,11,12]])
b = b.reshape((-1,1))

c = np.concatenate((b,a),axis=0)
print(c)
