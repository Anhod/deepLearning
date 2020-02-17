import numpy as np
import h5py
import matplotlib.pyplot as plt
import testCases
from dnn_utils import sigmoid,sigmoid_backward,relu,relu_backward
import lr_utils

#初始化参数(忘记了乘以0.01，以及断言)
# def initialize_parameters(n_x,n_h,n_y):
#     W1 = np.random.randn(n_h,n_x)*0.01
#     b1 = np.zeros((n_h,1))
#     W2 = np.random.randn(n_y,n_h)*0.01
#     b2=np.zeros((n_y,1))
#
#     assert (W1.shape == (n_h,n_x))
#     assert (b1.shape == (n_h,1))
#     assert (W2.shape == (n_y,n_h))
#     assert (b2.shape == (n_y,1))
#
#     parameters={"W1":W1,"W2":W2,"b1":b1,"b2":b2}
#     return parameters
#初始化权重矩阵以及偏向量
def initialize_parameters_deep(layer_dims):
    '''
    :param layers: 包含神经网络每一层的单元数
    :return:返回包含w和b的字典
    '''
    np.random.seed(3)
    parameters={}

    for i in range(1,len(layer_dims)):
        parameters["W"+str(i)] = np.random.rand(layer_dims[i],layer_dims[i-1]) / np.sqrt(layer_dims[i-1]) * 0.01
        parameters["b"+str(i)] = np.zeros((layer_dims[i],1))

        assert (parameters["W"+str(i)].shape == (layer_dims[i],layer_dims[i-1]))
        assert (parameters["b"+str(i)].shape == (layer_dims[i],1))

    return parameters

#前向传播
'''
前向传播：
    前向传播有以下三个步骤
    ·LINEAR:计算Z
    ·LINEAR - >ACTIVATION，其中激活函数将会使用ReLU或Sigmoid。        根据Z计算A，其中要选定激活函数
    ·[LINEAR - > RELU] ×（L-1） - > LINEAR - > SIGMOID（整个模型）
'''
def linear_forward(A,W,b):
    '''
    实施前向传播的LINEAR线性部分
    :param A: 来自上一层（或输入数据）的激活，维度为（上一层的节点数，示例的数量）
    :param W: 权重矩阵
    :param b: 偏向量
    :return:Z 激活功能的输入，也称为预激活参数
            cache：一个包含“A”、“W”和“b”的字典，存储这些变量以有效的计算后向传递
    '''
    Z = np.dot(W,A)+b

    assert Z.shape == (W.shape[0],A.shape[1])
    cache = (A,W,b)

    return Z,cache

def linear_activation_forward(A_prev,W,b,activation):
    '''
    实现LINEAR_ACTIVATION这一层的前向传播
    :param A_prev: 来自上一层（或输入层）的激活，维度为（上一层的节点数量，示例数）
    :param W: 权重矩阵
    :param b:偏向量
    :param activation:选择在此层中使用的激活函数，字符串类型，【“sigmoid” | “relu”】
    :return:A- 激活函数的输出，激活后的值
            cache，一个包含“linear_cache”和“activation_cache”的字典，我们需要存储它以有效的计算后向传播
    '''
    if activation == "sigmoid":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = sigmoid(Z)     #返回的是A和Z
    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)       # 返回的是A和Z

    assert (A.shape == (W.shape[0],A_prev.shape[1]))
    cache = (linear_cache,activation_cache)       #Z的计算式右边的因子A_prev,W,b      A的计算式右边的因子Z

    return A,cache

'''
多层模型的前向传播计算模型如下：
    AL即为yhat
'''
def L_model_forward(X,parameters):
    '''
    实现[LINEAR - > RELU] ×（L-1） - > LINEAR - > SIGMOID（整个模型）计算前向传播，也就是多层网络的前向传播，为后面每一层都执行LINEAR和ACTIVATION
    :param X: 数据，numpy数组，维度为（数入节点数量，示例数量）
    :param parameters: initialize_parameters_deep()的输出
    :return:
        AL:最后的激活值
        caches：包含以下内容的缓存列表：
                linear_relu_forward（）的每个cache（有L-1个，索引为从0到L-2）
                linear_sigmoid_forward（）的cache（只有一个，索引为L-1）
    '''
    caches=[]
    A=X
    L=len(parameters) // 2

    for i in range(1,L):
        A_prev=A
        A,cache = linear_activation_forward(A_prev,parameters["W"+str(i)],parameters["b"+str(i)],"relu")
        caches.append(cache)              #包含每一步的A  W  b，Z

    #计算yhat，利用sigmoid函数
    AL,cache = linear_activation_forward(A,parameters["W"+str(L)],parameters["b"+str(L)],"sigmoid")
    caches.append(cache)

    assert (AL.shape == (1,X.shape[1]))
    return AL,caches

#计算成本函数
'''
计算成本
'''
def compute_cost(AL,Y):
    '''
    根据成本函数计算成本
    :param AL: yhat
    :param Y: 标签向量
    :return: 交叉熵成本
    '''
    m=Y.shape[1]
    cost = - np.sum(np.multiply(np.log(AL),Y) + np.multiply(np.log(1-AL),1-Y)) / m
    cost = np.squeeze(cost)

    assert cost.shape == ()

    return cost

#后向传播
'''
反向传播：
    用于计算相对于参数的损失函数的梯度
    与前向传播类似，我们有需要使用三个步骤来构建反向传播：
        ·LINEAR 后向计算
        ·LINEAR -> ACTIVATION 后向计算，其中ACTIVATION 计算Relu或者Sigmoid 的结果
        ·[LINEAR -> RELU] ×\times× (L-1) -> LINEAR -> SIGMOID 后向计算 (整个模型)
'''
#计算单层的dW  db   dA
def linear_backward(dz,cache):
    '''
    为单层实现反向传播的先行部分（第L层）
    :param dz: 先后对于（当前第l层）线性输出的成本梯度
    :param cache: 来自当前层前向传播的值的元组（A_prev,W,b）
    :return:
        dA_prev - 相对于激活（前一层l-1）的成本梯度，与A_prev维度相同
         dW - 相对于W（当前层l）的成本梯度，与W的维度相同
         db - 相对于b（当前层l）的成本梯度，与b维度相同
    '''
    A_prev , W ,b = cache
    m = A_prev.shape[1]          #示例数
    dW = np.dot(dz,A_prev.T) / m
    db = np.sum(dz,axis=1,keepdims=True) / m
    dA_prev = np.dot(W.T,dz)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db

def linear_activation_backward(dA,cache,activation="relu"):
    '''
    :param dA: 当前层l的激活后的梯度值
    :param cache: 我们存储的用于有效计算反向传播的值的元组（值为linear_cache，activation_cache）
    :param activation: 要在此层中使用的激活函数名，字符串类型，【"sigmoid" | "relu"】
    :return:
            dA_prev - 相对于激活（前一层l-1）的成本梯度值，与A_prev维度相同
            dW - 相对于W（当前层l）的成本梯度值，与W的维度相同
            db - 相对于b（当前层l）的成本梯度值，与b的维度相同
    '''
    linear_cache,activation_cache=cache
    if activation == "relu":
        dZ = relu_backward(dA,activation_cache)         #计算激活函数为"relu"的时候的dZ
        dA_prev,dW,db = linear_backward(dZ,linear_cache)
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)     #计算激活函数为"sigmoid"的时候的dZ
        dA_prev, dW, db = linear_backward(dZ, linear_cache)      #往前一层需要用到dA_prev

    return dA_prev,dW,db

def L_model_backward(AL, Y, caches):
    """
    对[LINEAR-> RELU] *（L-1） - > LINEAR - > SIGMOID组执行反向传播，就是多层网络的向后传播

    参数：
     AL - 概率向量，正向传播的输出（L_model_forward（））
     Y - 标签向量（例如：如果不是猫，则为0，如果是猫则为1），维度为（1，数量）
     caches - 包含以下内容的cache列表：
                 linear_activation_forward（"relu"）的cache，不包含输出层
                 linear_activation_forward（"sigmoid"）的cache

    返回：
     grads - 具有梯度值的字典
              grads [“dA”+ str（l）] = ...
              grads [“dW”+ str（l）] = ...
              grads [“db”+ str（l）] = ...
    """
    grads = {}
    L = len(caches)     #因为每一层的A  W  b，Z都有缓存（cache），所以这里为caches，得到cache的数量即得到层数（输入层不算）   L=4
    m = AL.shape[1]                 #示例数，209
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))       #最后一层的dA,由损失函数对a求导

    current_cache = caches[L - 1]

    #最后一层的dA，dW,db,为什么要分开计算，因为dA的计算要用到后一层的W和dZ，所以不放在循环里，且激活函数也与其他的层不一样
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache,"sigmoid")   #这里算出来的应该是前一层的dA

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, "relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


'''
更新参数
'''
def update_parameters(parameters, grads, learning_rate):
    """
    使用梯度下降更新参数

    参数：
     parameters - 包含你的参数的字典
     grads - 包含梯度值的字典，是L_model_backward的输出

    返回：
     parameters - 包含更新参数的字典
                   参数[“W”+ str（l）] = ...
                   参数[“b”+ str（l）] = ...
    """
    L = len(parameters) // 2  # 整除
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate=0.0075, num_iterations=3000, print_cost=False, isPlot=True):
    '''
    实现一个L层神经网络：[LINEAR-> RELU] *（L-1） - > LINEAR-> SIGMOID。
    :param X: 输入的数据，维度为(n_x，例子数)
    :param Y: 标签，向量，0为非猫，1为猫，维度为(1,数量)
    :param layers_dims: 层数的向量，维度为(n_y,n_h,···,n_h,n_y)
    :param learning_rate: 学习率
    :param num_iterations: 迭代的次数
    :param print_cost: 是否打印成本值，每100次打印一次
    :param isPlot: 是否绘制出误差值的图谱
    :return: 模型学习的参数。 然后他们可以用来预测。
    '''
    np.random.seed(1)
    costs = []

    parameters = initialize_parameters_deep(layers_dims)                            #先对参数进行初始化

    for i in range(0, num_iterations):
        AL, caches = L_model_forward(X, parameters)

        cost = compute_cost(AL, Y)

        grads = L_model_backward(AL, Y, caches)

        parameters = update_parameters(parameters, grads, learning_rate)

        # 打印成本值，如果print_cost=False则忽略
        if i % 100 == 0:
            # 记录成本
            costs.append(cost)
            # 是否打印成本值
            if print_cost:
                print("第", i, "次迭代，成本值为：", np.squeeze(cost))
    # 迭代完成，根据条件绘制图
    if isPlot:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    return parameters

train_set_x_orig , train_set_y , test_set_x_orig , test_set_y , classes = lr_utils.load_dataset()

train_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

train_x = train_x_flatten / 255
train_y = train_set_y
test_x = test_x_flatten / 255
test_y = test_set_y

layers_dims = [12288, 20, 7, 5, 1]          #  5-layer model，第一位数字为输入的特征数   64x64x3
parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500, print_cost = True,isPlot=True)