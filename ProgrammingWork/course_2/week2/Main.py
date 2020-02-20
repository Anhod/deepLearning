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
'''
In practice, you'll often get faster results if you do not use neither the whole training set, 
nor only one training example, to perform each update. Mini-batch gradient descent uses 
an intermediate number of examples for each step. With mini-batch gradient descent, 
you loop over the mini-batches instead of looping over individual training examples.
'''

#-----------------------------------根据小批量的大小对数据集进行随机分组-----------------------------------
def random_mini_batches(X,Y,mini_batch_size=64,seed=0):
    np.random.seed(seed)

    m = X.shape[1]
    mini_batches=[]

    #对原有数据集和标签向量进行重新分组
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))

    num_complete_minibatches = math.floor(m / mini_batch_size)
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[:, mini_batch_size*k : mini_batch_size*(k+1)]
        mini_batch_Y = shuffled_Y[:, mini_batch_size*k : mini_batch_size*(k+1)]

        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)

    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:,num_complete_minibatches * mini_batch_size : ]
        mini_batch_Y = shuffled_Y[:,num_complete_minibatches * mini_batch_size : ]

        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches

X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)

print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))