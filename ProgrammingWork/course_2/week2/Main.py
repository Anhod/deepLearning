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

#根据小批量的大小对数据集进行随机分组
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

#-------------------------------------------动量优化算法-------------------------------------------
'''
Because mini-batch gradient descent makes a parameter update after seeing just a subset of examples, 
the direction of the update has some variance, and so the path taken by mini-batch gradient descent will "oscillate" 
toward convergence. Using momentum can reduce these oscillations.

开始动量梯度下降
'''
#对Vdw,Vdb进行初始化，它的对数与层数相同（看公式即可得知）
def initialize_velocity(prameters):
    L = len(parameters) // 2
    v = {}

    for i in range(L):
        v["dW"+str(i+1)] = np.zeros((parameters["W"+str(i+1)].shape[0],parameters["W"+str(i+1)].shape[1]))
        v["db"+str(i+1)] = np.zeros((parameters["b"+str(i+1)].shape[0],parameters["b"+str(i+1)].shape[1]))

    return v

#利用动量来更新梯度
def update_parameters_with_momentum(parameters,grads,v,beta,learning_rate):
    L = len(parameters) // 2

    for l in range(L):
        v["dW" + str(l+1)] = beta * v["dW" + str(l+1)] + (1-beta) * grads["dW" + str(l+1)]
        v["db" + str(l+1)] = beta * v["db" + str(l+1)] + (1-beta) * grads["db" + str(l+1)]

        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] -learning_rate * v["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] -learning_rate * v["db" + str(l+1)]

    return parameters,v
'''
Note that:
    ->The velocity is initialized with zeros. So the algorithm will take a few iterations to "build up" velocity and 
     start to take bigger steps.
    ->If  β=0 , then this just becomes standard gradient descent without momentum.
'''

#-------------------------------------------Adam优化算法-------------------------------------------
def initialize_Adam(parameters):
    L =len(parameters) // 2
    v = {}
    s = {}

    for i in range(L):
        v["dW"+str(i+1)] = np.zeros((parameters["W" + str(i+1)].shape[0],parameters["W" + str(i+1)].shape[1]))
        v["db"+str(i+1)] = np.zeros((parameters["b" + str(i+1)].shape[0],parameters["b" + str(i+1)].shape[1]))

        s["dW"+str(i+1)] = np.zeros((parameters["W" + str(i+1)].shape[0],parameters["W" + str(i+1)].shape[1]))
        s["db"+str(i+1)] = np.zeros((parameters["b" + str(i+1)].shape[0],parameters["b" + str(i+1)].shape[1]))

    return v,s

#使用Adam来更新参数
def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,beta1=0.9,beta2=0.999,epsilon=1e-8):
    """
       Update parameters using Adam

       Arguments:
       parameters -- python dictionary containing your parameters:
                       parameters['W' + str(l)] = Wl
                       parameters['b' + str(l)] = bl
       grads -- python dictionary containing your gradients for each parameters:
                       grads['dW' + str(l)] = dWl
                       grads['db' + str(l)] = dbl
       v -- Adam variable, moving average of the first gradient, python dictionary
       s -- Adam variable, moving average of the squared gradient, python dictionary
       learning_rate -- the learning rate, scalar.
       beta1 -- Exponential decay hyperparameter for the first moment estimates
       beta2 -- Exponential decay hyperparameter for the second moment estimates
       epsilon -- hyperparameter preventing division by zero in Adam updates

       Returns:
       parameters -- python dictionary containing your updated parameters
       v -- Adam variable, moving average of the first gradient, python dictionary
       s -- Adam variable, moving average of the squared gradient, python dictionary
       """
    L = len(parameters) // 2

    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        v["dW" + str(l + 1)] = beta1 * v["dW" + str(l + 1)] + (1-beta1) * grads["dW" + str(l + 1)]
        v["db" + str(l + 1)] = beta1 * v["db" + str(l + 1)] + (1-beta1) * grads["db" + str(l + 1)]
        v_corrected["dW" + str(l + 1)] = v["dW" + str(l + 1)] / (1 - np.power(beta1,t))
        v_corrected["db" + str(l + 1)] = v["db" + str(l + 1)] / (1 - np.power(beta1,t))

        s["dW" + str(l + 1)] = beta2 * s["dW" + str(l + 1)] + (1-beta2) * np.square(grads["dW" + str(l + 1)])
        s["db" + str(l + 1)] = beta2 * s["db" + str(l + 1)] + (1-beta2) * np.square(grads["db" + str(l + 1)])
        s_corrected["dW" + str(l + 1)] = s["dW" + str(l + 1)] / (1 - np.power(beta2,t))
        s_corrected["db" + str(l + 1)] = s["db" + str(l + 1)] / (1 - np.power(beta2,t))

        #更新参数
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * (v_corrected["dW" + str(l + 1)] / np.sqrt(s_corrected["dW" + str(l + 1)] + epsilon))
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * (v_corrected["db" + str(l + 1)] / np.sqrt(s_corrected["db" + str(l + 1)] + epsilon))

    return parameters, v, s

#-------------------------------------------下面来构建模型-------------------------------------------
train_X,train_Y = load_dataset()

def model(X,Y,layers_dims,optimizer,learning_rate = 0.0007,mini_batch_size=64,beta=0.9,beta1=0.9,beta2=0.999,epsilon=1e-8,
          num_epochs=10000,print_cost=True):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    m = X.shape[1]

    parameters = initialize_parameters(layers_dims)

    #初始化优化算法
    if optimizer=="gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v,s = initialize_Adam(parameters)

    #开始训练
    for i in range(num_epochs):
        seed = seed + 1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)
        cost_total = 0

        for minibatch in minibatches:
            (minibatch_X,minibatch_Y) = minibatch

            #前向传播
            a3,caches = forward_propagation(minibatch_X,parameters)

            #计算损失函数
            cost_total += compute_cost(a3,minibatch_Y)

            #反向传播
            grads = backward_propagation(minibatch_X,minibatch_Y,caches)

            #更新参数
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters,learning_rate,grads)
            elif optimizer == "momentum":
                parameters,v = update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer == "adam":
                t = t + 1
                parameters,v,s = update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)

        costavg = cost_total / m

        if print_cost and i % 1000==0:
            print("Cost after epoch %i:%f" % (i,costavg))
            costs.append(costavg)
        if print_cost and i % 100==0:
            costs.append(costavg)

                # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def model1(X, Y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours
    m = X.shape[1]  # number of training examples

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(X, Y, mini_batch_size, seed)
        cost_total = 0

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost and add to the cost total
            cost_total += compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, learning_rate, grads)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v, s = update_parameters_with_adam(parameters, grads, v, s,
                                                               t, learning_rate, beta1, beta2, epsilon)
        cost_avg = cost_total / m

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost_avg))
        if print_cost and i % 100 == 0:
            costs.append(cost_avg)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters
#Mini-batch Gradient descent
# train 3-layer model
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "gd")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)