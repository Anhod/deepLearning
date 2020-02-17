import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils_v2 import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

#%matplotlib inline
'''
设置一个固定的随机种子，以保证接下来的步骤中我们的结果是一致的
seed值不一样的时候，点的坐标是不一样的，所以生成的图也会有一点区别
'''
np.random.seed(1)

'''
接下来，我们来看一下我们将要使用的数据集，下面的代码会将一个花的2类数据集加载到变量X和Y中
'''
X,Y=load_planar_dataset()

'''
使用matplotlib可视化数据集
数据看起来像一朵花（y=0）和一些蓝色（y=1）的数据的点的花朵的图案
我们的目标是建立一个模型来适应这些数据

现在，我们有了一下的东西：
    -X：一个numpy的矩阵，包含了这些数据点的数值（类似坐标）（红点、蓝点各两百个）
    -Y：一个numpy的向量，对应着的是X的标签【0|1】（红色：0，蓝色：1）
'''
plt.scatter(X[0,:],X[1,:],c=np.squeeze(Y),s=40,cmap=plt.cm.Spectral)        #绘制散点图，函数中c表示颜色
plt.show()

#我们来仔细地看看这些数据
shape_X=X.shape
shape_Y=Y.shape
m=Y.shape[1]

print("X的维度为："+str(shape_X))
print("Y的维度为："+str(shape_Y))
print("数据集里的数据有："+str(m)+"个")

'''
在构建完整的神经网络之前，先看看这个问题在逻辑回归上的表现如何，我们可以使用
sklearn的内置函数来做到这一点

准确性只有47%的原因是数据集不是线性可分的，所以逻辑回归表现不佳，
'''
clf=sklearn.linear_model.LogisticRegressionCV()
clf.fit(X.T,Y.T)

#把逻辑分类器的分类绘制出来：
plot_decision_boundary(lambda x:clf.predict(x),X,Y)
plt.title("Logistic Regression")
LR_predictions=clf.predict(X.T)
print("逻辑回归的准确性：%d "%float((np.dot(Y,LR_predictions)+np.dot(1-Y,1-LR_predictions))/float(Y.size)*100)+"%"+"(正确标记的数据点所占的百分比)")

'''
构建神经网络：
    1.定义神经网络
    2.初始化模型的参数
    3.循环：
        实施前向传播
        计算损失
        实现向后传播
        更新参数（梯度下降）
'''
#定义神经网络的结构：
def layer_size(X,Y):
    '''
    :param X: 输入数据集，维度为（输入的数量，训练/测试例子的数量）
    :param Y:标签，维度为（输出的数量，训练/测试例子的数量）
    :return n_x  n_h  n_y
    '''
    n_x=X.shape[0]                  #输入层的神经元数量
    n_h=4                           #隐藏层的神经元数量
    n_y=Y.shape[0]                  #输出层的神经元数量

    return n_x,n_h,n_y
print("=====测试layer_size=====")
x_assess,y_assess=layer_sizes_test_case()
n_x,n_h,n_y=layer_size(x_assess,y_assess)
print("输入层的神经元个数为："+str(n_x))
print("隐藏层的神经元个数为："+str(n_h))
print("输出层的神经元个数为："+str(n_y)+"\n")

'''
初始化参数，包括w[1]、b[1]、w[2]、b[2]
'''
#初始化权重矩阵以及偏移量
def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    w1=np.random.rand(n_h,n_x)*0.01     #行数是神经元的个数，列数是输入特征数
    b1=np.zeros((n_h,1))

    w2=np.random.rand(n_y,n_h)*0.01
    b2=np.zeros((n_y,1))

    assert w1.shape == (n_h,n_x)
    assert b1.shape == (n_h,1)
    assert w2.shape == (n_y,n_h)
    assert b2.shape == (n_y,1)

    parameters={"w1":w1,"w2":w2,"b1":b1,"b2":b2}
    return parameters

print("=====测试initialize_parameters=====")
n_x,n_h,n_y=initialize_parameters_test_case()
parameters=initialize_parameters(n_x,n_h,n_y)
print("w1为："+str(parameters["w1"]))
print("b1为："+str(parameters["b1"]))
print("w2为："+str(parameters["w2"]))
print("b2为："+str(parameters["b2"]))

'''
循环
    前向传播:需要参数，X矩阵，权重矩阵以及偏移量，激活函数
'''
def forward_propagation(X,parameters):
    '''
    :param X:维度为(n_x,m)的向量
    :param parameters: 包括权重矩阵以及偏向量初始后的值
    :return:
    '''
    w1=parameters["w1"]
    b1=parameters["b1"]
    w2=parameters["w2"]
    b2=parameters["b2"]

    z1=np.dot(w1,X)+b1
    A1=np.tanh(z1)
    z2=np.dot(w2,A1)+b2
    A2=sigmoid(z2)

    assert (A2.shape == (1,X.shape[1]))
    cache={"z1":z1,"A1":A1,"z2":z2,"A2":A2}

    return (A2,cache)

print("\n=====测试forward_propagation=====")
x_assess,parameters=forward_propagation_test_case()
A2,cache=forward_propagation(x_assess,parameters)
print(np.mean(cache["z1"]), np.mean(cache["A1"]), np.mean(cache["z2"]), np.mean(cache["A2"]))

'''
    计算损失
'''
def compute_cost(A2,Y,parameters):
    '''
        计算交叉熵成本
    :param A2:使用sigmoid()函数计算的第二次激活后的数值
    :param Y:"True"标签向量,维度为（1，数量）
    :param parameters:一个包含W1，B1，W2和B2的字典类型的变量
    :return:成本
    '''
    m=Y.shape[1]
    w1=parameters["w1"]
    w2=parameters["w2"]

    #计算成本
    logprobs=np.multiply(np.log(A2),Y)+np.multiply((1-Y),np.log(1-A2))
    cost=-(np.sum(logprobs)/m)
    cost=float(np.squeeze(cost))

    assert (isinstance(cost, float))

    return cost

#测试compute_cost
print("\n=========================测试compute_cost=========================")
A2 , Y_assess , parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2,Y_assess,parameters)))

'''
    反向传播，计算导数等等 
'''
def backward_propagation(parameters,cache,X,Y):
    '''
    :param parameters: 包含我们的参数的一个字典类型的变量。
    :param cache: 包含“Z1”，“A1”，“Z2”和“A2”的字典类型的变量。
    :param X: 输入数据，维度为（2，数量）
    :param Y: “True”标签，维度为（1，数量）
    :return:
    '''

    m=Y.shape[1]
    w1=parameters["w1"]
    w2=parameters["w2"]
    A1=cache["A1"]
    A2=cache["A2"]

    dZ2=A2-Y
    dW2=(1/m)*(np.dot(dZ2,A1.T))
    db2=(1/m)*np.sum(dZ2,axis=1,keepdims=True)

    dZ1=np.multiply(np.dot(w2.T,dZ2),1-np.power(A1,2))
    dW1=(1/m)*(np.dot(dZ1,X.T))
    db1=(1/m)*np.sum(dZ1,axis=1,keepdims=True)

    grads={"dW1":dW1,"dW2":dW2,"db1":db1,"db2":db2}
    return grads

#测试backward_propagation
print("\n=========================测试backward_propagation=========================")
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()
grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))

'''
我们需要使用(dW1, db1, dW2, db2)来更新(W1, b1, W2, b2)，之后再返回更新了的(W1, b1, W2, b2)
'''
def update_parameters(parameters,grads,learning_rate=1.2):
    w1=parameters["w1"]
    w2=parameters["w2"]
    b1=parameters["b1"]
    b2=parameters["b2"]

    dW1=grads["dW1"]
    dW2=grads["dW2"]
    db1=grads["db1"]
    db2=grads["db2"]

    w1=w1-learning_rate*dW1
    w2=w2-learning_rate*dW2
    b1=b1-learning_rate*db1
    b2=b2-learning_rate*db2

    parameters={"w1":w1,"w2":w2,"b1":b1,"b2":b2}
    return parameters

#测试update_parameters
print("\n=========================测试update_parameters=========================")
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["w1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["w2"]))
print("b2 = " + str(parameters["b2"]))

'''
我们现在把上面的东西整合到nn_model()中，神经网络模型必须以正确的顺序使用先前的功能。
'''
def nn_model(X,Y,n_h,num_iterations,print_cost=False):
    '''
    :param X:数据集,维度为（2，示例数）
    :param Y:标签，维度为（1，示例数）
    :param n_h:隐藏层的数量
    :param num_iterations: 梯度下降循环中的迭代次数
    :param print_cost:如果为True，则每1000次迭代打印一次成本数值
    :return: parameters - 模型学习的参数，它们可以用来进行预测。
    '''

    np.random.seed(3)
    n_x=layer_size(X,Y)[0]
    n_y=layer_size(X,Y)[2]

    #得到初始化的权重矩阵 以及 偏移量
    parameters=initialize_parameters(n_x,n_h,n_y)
    w1=parameters["w1"]
    w2=parameters["w2"]
    b1=parameters["b1"]
    b2=parameters["b2"]

    for i in range(num_iterations):
        A2,cache=forward_propagation(X,parameters)                  #前向传播
        cost=compute_cost(A2,Y,parameters)                          #计算损失
        grads=backward_propagation(parameters,cache,X,Y)            #反向传播
        parameters=update_parameters(parameters,grads,learning_rate=0.5)        #更新参数

        if print_cost:
            if i%1000==0:
                print("第",i,"次循环，成本为："+str(cost))

    return parameters

#测试nn_model
print("=========================测试nn_model=========================")
X_assess, Y_assess = nn_model_test_case()

parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=False)
print("W1 = " + str(parameters["w1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["w2"]))
print("b2 = " + str(parameters["b2"]))


def predict(parameters, X):
    """
    使用学习的参数，为X中的每个示例预测一个类

    参数：
		parameters - 包含参数的字典类型的变量。
	    X - 输入数据（n_x，m）

    返回
		predictions - 我们模型预测的向量（红色：0 /蓝色：1）

     """
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)

    return predictions

parameters = nn_model(X, Y, n_h = 4, num_iterations=10000, print_cost=True)

#绘制边界
plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
plt.show()

'''
    更改隐藏节点的数量
'''
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50] #隐藏层数量

for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    parameters = nn_model(X, Y, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    predictions = predict(parameters, X)
    accuracy = float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100)
    print ("隐藏层的节点数量： {}  ，准确率: {} %".format(n_h, accuracy))
plt.savefig("不同隐藏节点的模型")
plt.show()