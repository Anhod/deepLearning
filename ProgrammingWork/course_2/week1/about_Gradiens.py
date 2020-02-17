import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import gc_utils       #第三部分，梯度检验
import testCases

def forward_propagation(X,Y,parameters):
   '''
    实现前向传播，并计算成本

   :param X: 训练集
   :param Y: 标签向量
   :param parameters:包含权重矩阵和偏向量的字典
   :return:
      cost 成本函数
   '''
   m = X.shape[1]
   W1 = parameters["W1"]
   b1 = parameters["b1"]
   W2 = parameters["W2"]
   b2 = parameters["b2"]
   W3 = parameters["W3"]
   b3 = parameters["b3"]

    #前向传播
   Z1 = np.dot(W1,X) + b1
   A1 = gc_utils.relu(Z1)

   Z2 = np.dot(W2,A1) + b2
   A2 = gc_utils.relu(Z2)

   Z3 = np.dot(W3,A2) + b3
   A3 = gc_utils.sigmoid(Z3)

    #计算成本函数
   logprobs = np.multiply(-np.log(A3),Y) + np.multiply(-np.log(1-A3),1-Y)
   cost = (1 / m) * np.sum(logprobs)
   cache = (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3)

   return cost,cache

def backward_propagation(X,Y,cache):
    '''
    实现反向传播

    :param X: 输入数据集
    :param Y: 标签向量
    :param cache: 来自前向传播的输出
    :return:
        gradients：包含每个参数、激活和预激活前变量相关的成本梯度
    '''
    m = X.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = (1/m) *(A3 - Y)
    dW3 = np.dot(dZ3,A2.T)
    db3 = np.sum(dZ3,axis=1,keepdims=True)

    dA2 = np.dot(W3.T,dZ3)
    dZ2 = (1/m) * np.multiply(dA2,np.int64(A2>0))
    dW2 = np.dot(dZ2,A1.T)
    db2 = np.sum(dZ2,axis=1,keepdims=True)

    dA1 = np.dot(W2.T,dZ2)
    dZ1 = (1/m) * np.multiply(dA1,np.int64(A1>0))
    dW1 = np.dot(dZ1,X.T)
    db1 = np.sum(dZ1,axis=1,keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,
                 "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
                 "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

def gradient_check(parameters,gradients,X,Y,epsilon=1e-7):
    '''
    检查后向传播是否正确计算了前向传播输出的成本梯度

    :param parameters: 包含了权重矩阵和偏置向量的字典
    :param gradients: 包含与参数相关的成本梯度
    :param X: 输入数据点
    :param Y: 标签向量
    :param epsilon:计算输入微小的微小偏移以计算近似梯度
    :return:
        difference：近似梯度和后向传播梯度之间的差异
    '''
    parameters_values,keys = gc_utils.dictionary_to_vector(parameters)          #向量之后的权重矩阵和偏向量
    grad = gc_utils.gradients_to_vector(gradients)                              #后向传播之后的梯度
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters,1))
    J_minus = np.zeros((num_parameters,1))
    gradapprox = np.zeros((num_parameters,1))

    for i in range(num_parameters):
        #计算J_plus
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] = thetaplus[i][0] + epsilon
        J_plus[i],cache = forward_propagation(X,Y,gc_utils.vector_to_dictionary(thetaplus))

        #计算J_minus
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        J_minus[i], cache = forward_propagation(X, Y, gc_utils.vector_to_dictionary(thetaminus))

        #计算gradapprox
        gradapprox[i] = (J_plus[i] - J_minus[i])/(2 * epsilon)

    #计算差异，然后进行比较
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    differece = numerator / denominator

    if differece < epsilon:
        print("梯度检查：梯度正常！")
    else:
        print("梯度检查：梯度超出阈值！")

    return differece

X,Y,parameters = testCases.gradient_check_n_test_case()
cost,cache = forward_propagation(X,Y,parameters)
gradients = backward_propagation(X,Y,cache)
difference = gradient_check(parameters,gradients,X,Y,epsilon=1e-7)
print("差异值为：",difference)