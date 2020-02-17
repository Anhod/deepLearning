import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import reg_utils        #第二部分，正则化

'''
我们要做以下三件事，来对比出不同的模型的优劣：
1.不使用正则化
2.使用正则化
    2.1 使用L2正则化
    2.2 使用随机节点删除
'''
train_X,train_Y,test_X,test_Y=reg_utils.load_2D_dataset(is_plot=True)

'''
我们来看一下我们的模型：
    正则化模式 - 将lambd输入设置为非零值。 我们使用“lambd”而不是“lambda”，因为“lambda”是Python中的保留关键字。
    随机删除节点 - 将keep_prob设置为小于1的值
'''
def model(X,Y,learning_rate=0.3,num_iterations=30000,print_cost=True,is_plot=True,lambd=0,keep_prob=1):
    '''
    实现一个三层的神经网络

    :param X: 输入的数据
    :param Y: 标签
    :param learning_rate: 学习率
    :param num_iterations: 迭代次数
    :param print_cost: 是否打印成本
    :param is_plot: 是否绘制梯度下降的曲线图
    :param lambd: 正则化的超参数
    :param keep_prob: 随即删除节点的概率
    :return:
        parameters：学习后的参数
    '''
    grads={}
    costs=[]
    m = X.shape[1]
    layers_dim=[X.shape[0],20,3,1]

    #初始化参数
    parameters = reg_utils.initialize_parameters(layers_dim)

    #开始学习
    for i in range(0,num_iterations):
        #前向传播,是否随机删除节点
        if keep_prob == 1:
            a3,cache = reg_utils.forward_propagation(X,parameters)
        elif keep_prob < 1:
            a3,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
        else:
            print("keep_prob参数错误！程序退出。")
            exit

        #计算成本，是否使用二范数
        if lambd == 0:
            cost = reg_utils.compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)

        #反向传播，可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用
        assert (lambd == 0 or keep_prob == 1)

        ##两个参数的使用情况
        if(lambd == 0 and keep_prob == 1):
            #两种正则化都不使用
            grads = reg_utils.backward_propagation(X,Y,cache)
        elif lambd != 0:
            #使用L2正则化
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob < 1:
            #使用随即失活正则化
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)

        #更新参数
        parameters = reg_utils.update_parameters(parameters,grads,learning_rate)

        #记录并打印成本
        if i % 1000 == 0:
            costs.append(cost)
            if(print_cost and i % 10000 == 0):
                print("第"+str(i)+"次迭代，成本值为："+str(cost))

    #是否绘制成本函数
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(x1,000')
        plt.title("Learning rate = "+str(learning_rate))
        plt.show()

    return parameters

#------------------------------不使用正则化------------------------------
parameters = model(train_X,train_Y,is_plot=True)
print("训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

#很明显的过拟合现象
plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)



#------------------------------使用正则化------------------------------
#使用L2正则化，适当修改损失函数
'''
λ的值是可以使用开发集调整时的超参数。L2正则化会使决策边界更加平滑。
如果λ太大，也可能会“过度平滑”，从而导致模型高偏差。
L2正则化实际上在做什么？L2正则化依赖于较小权重的模型比具有较大权重的模型更简单这样的假设，
因此，通过削减成本函数中权重的平方值，可以将所有权重值逐渐改变到到较小的值。
权值数值高的话会有更平滑的模型，其中输入变化时输出变化更慢，但是你需要花费更多的时间。L2正则化对以下内容有影响：
    ·成本计算       ： 正则化的计算需要添加到成本函数中
    ·反向传播功能     ：在权重矩阵方面，梯度计算时也要依据正则化来做出相应的计算
    ·重量变小（“重量衰减”) ：权重被逐渐改变到较小的值。
'''
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    '''
    实现了我们添加了L2正则化的模型的后向传播

    :param A3: 正向传播的输出结果，yhat
    :param Y: 标签向量
    :param parameters:包含模型学习后的参数的字典
    :param lambd:
    :return:
        cost：是用公式2计算出来的正则化损失的值
    '''
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = reg_utils.compute_cost(A3,Y)

    L2_regularization_cost = lambd *(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3))) / (2*m)

    cost = cross_entropy_cost + L2_regularization_cost

    return cost

def backward_propagation_with_regularization(X,Y,cache,lambd):
    '''
    实现我们添加了L2正则化的模型的后向传播

    :param X: 输入数据集
    :param Y: 标签向量
    :param cache: 来自前向传播的cache输出
    :param lambd: 超参数
    :return:
        gradients：一个包含了每个参数、激活值和预激活值变量的梯度的字典
    '''
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
    db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

parameters = model(train_X,train_Y,lambd = 0.7,is_plot=True)
print("使用正则化，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用正则化，测试集:")
predictions_test = reg_utils.predict(test_X, test_Y, parameters)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)


#------------------------------随机失活------------------------------
def forward_propagation_with_dropout(X,parameters,keep_prob=0.5):
    '''
    实现具有随机舍弃节点的前向传播

    :param X: 输入数据集
    :param parameters:包含W1,W2,W3,b1,b2,b3的字典，第一次是随机初始化之后的值，之后为迭代学习一次之后的值
    :param keep_prob: 随机删除的概率
    :return:
        A3，最后的激活值
        cache：存储了一些用于计算反向传播的数值的元组
    '''
    np.random.seed(1)

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1,X) + b1
    A1 = reg_utils.relu(Z1)

    #下面的步骤1-4对应于上述的步骤
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1* D1
    A1 = A1 / keep_prob

    Z2 = np.dot(W2, A1) + b2
    A2 = reg_utils.relu(Z2)

    # 下面的步骤1-4对应于上述的步骤
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob

    Z3 = np.dot(W3,A2) + b3
    A3 = reg_utils.sigmoid(Z3)

    cache = (Z1,D1,A1,W1,b1,Z2,D2,A2,W2,b2,Z3,A3,W3,b3)
    return A3,cache

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    '''
    实现我们随机删除模型的后向传播

    :param X: 输入数据集
    :param Y: 标签向量
    :param cache: 来自前向传播（随机删除）后的cache输出
    :param keep_prob: 随机删除的概率，实数
    :return:
        gradients：一个关于每个参数、激活值和预激活值变量的梯度值的字典
    '''
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - Y
    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2                  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA2 = dA2 / keep_prob           # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
    dA1 = dA1 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值

    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, X.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}

    return gradients

parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3,is_plot = True)

print("使用随机删除节点，训练集:")
predictions_train = reg_utils.predict(train_X, train_Y, parameters)
print("使用随机删除节点，测试集:")
reg_utils.predictions_test = reg_utils.predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75, 0.40])
axes.set_ylim([-0.75, 0.65])
reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), train_X, train_Y)