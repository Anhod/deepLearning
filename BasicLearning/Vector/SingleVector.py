import numpy as np
import time as time

#运用向量提高运行速度
#运用向量
a=np.random.rand(1000000)
b=np.random.rand(1000000)

tic=time.time()
c=np.dot(a,b)
toc=time.time()

print("Vectorized version:"+str(1000*(toc-tic))+"ms")
print(c)

#不运用向量
c=0
tic=time.time()

for i in range(1000000):
    c+=a[i]*b[i]
toc=time.time()

print(c)
print("for loop the times is:"+str(1000*(toc-tic))+"ms")