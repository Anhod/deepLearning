import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads,initialize_parameters,forward_propagation,backward_propagation
from opt_utils import compute_cost,predict,plot_decision_boundary,predict_dec,load_dataset,sigmoid,relu
from testCase import *

plt.rcParams["figure.figsize"] = (7.0,4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#---------------------------------------第一部分，梯度下降---------------------------------------
def update_parameters_with_gd(parameters,learning_rate,grads):
    '''
    使用梯度下降更新参数

    :param parameters: 包含W,b的字典
    :param learning_rate: 学习率
    :param grads: 使用成本计算出来的梯度
    :return:
    '''

    L = (len(parameters)) // 2

    for l in range(L):
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate * grads["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate * grads["db"+str(l+1)]

    return parameters


parameters,grads,learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters,learning_rate,grads)
print("W1 =\n" + str(parameters["W1"]))
print("b1 =\n" + str(parameters["b1"]))
print("W2 =\n" + str(parameters["W2"]))
print("b2 =\n" + str(parameters["b2"]))

'''
A variant of this is Stochastic Gradient Descent (SGD，随机梯度下降), which is equivalent to mini-batch gradient descent 
where each mini-batch has just 1 example. The update rule that you have just implemented does not change. 
What changes is that you would be computing gradients on just one training example at a time, rather than 
on the whole training set. The code examples below illustrate the difference between stochastic gradient descent 
and (batch) gradient descent. 

----------------------------**(Batch) Gradient Descent**:
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost += compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)
        
```

---------------------------- **Stochastic Gradient Descent**:
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost += compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)
```

'''