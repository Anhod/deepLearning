import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import init_utils       #第一部分，初始化

#%matplotlib inline
#设置图像的细节
plt.rcParams['figure.figsize']=(7.0,4.0)                #设置图像大小
plt.rcParams['image.interpolation']='nearest'           #这个是指定插值的方式，图像缩放之后，肯定像素要进行重新计算的，就靠这个参数来指定重新计算像素的方式
plt.rcParams['image.cmap']='gray'                       #灰度空间

#首先来看看数据集
train_X,train_Y,test_X,test_Y = init_utils.load_dataset(is_plot=True)

def model(X,Y,learning_rate=0.01,num_iterations=15000,print_cost=True,initialization="he",is_plot=True):
    """
    实现一个三层的神经网络

    :param X: 输入的数据，维度为（2，要训练/测试的数量）
    :param Y: 标签，【0|1】，维度为（1，对应的是输入的数据的标签）
    :param learning_rate: 学习速率
    :param num_iterations: 迭代的次数
    :param print_cost: 是否打印成本值
    :param initialization: 字符串类型，初始化的类型【“zeros”|“random”|“he”】
    :param is_plot: 是否绘制梯度下降的曲线图
    :return:
        parameters，学习后的参数
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dim=[X.shape[0],10,5,1]

    #选择初始化参数的类型
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dim)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dim)
    elif initialization == "he":
        parameters = initialize_parameters_he(layers_dim)
    else :
        print("错误的初始化参数！程序退出")
        exit()

    #开始学习
    for i in range(0,num_iterations):
        #前向传播
        a3,cache = init_utils.forward_propagation(X,parameters)

        #计算成本
        cost = init_utils.compute_loss(a3,Y)

        #反向传播
        grads = init_utils.backward_propagation(X,Y,cache)

        #更新参数
        parameters = init_utils.update_parameters(parameters,grads,learning_rate)

        #记录成本
        if i%1000 == 0:
            costs.append(cost)
            if print_cost:
                print("第"+str(i)+"次迭代，成本值为："+str(cost))

    #学习完毕，绘制成本函数
    if is_plot:
        plt.plot(costs)
        plt.ylabel('cost')
        plt.xlabel('iterations(per hundreds)')
        plt.title("learning rate ="+str(learning_rate))
        plt.show()
    return parameters

#-----------------------------------------------零初始化-----------------------------------------------
def initialize_parameters_zeros(layers_dim):
    """
    将模型的参数全部设置为0
    :param layers_dim:
    :return:
    """
    parameters={}

    for i in range(1,len(layers_dim)):
        parameters["W"+str(i)]=np.zeros((layers_dim[i],layers_dim[i-1]))
        parameters["b"+str(i)]=np.zeros((layers_dim[i],1))

        assert parameters["W"+str(i)].shape == (layers_dim[i],layers_dim[i-1])
        assert parameters["b"+str(i)].shape == (layers_dim[i],1)

    return parameters

#可以看出，将矩阵初始化为0的时候，根本没学习
parameters = model(train_X,train_Y,initialization="zeros",is_plot=True)

print("训练集")
predictions_train = init_utils.predict(train_X,train_Y,parameters)
print("测试集")
predictions_test = init_utils.predict(train_X,train_Y,parameters)

#上述性能确实很差，而且成本并没有真正降低，算法的性能也比随即猜测要好，看看预测和决策边界的细节
print("predictions_train = "+str(predictions_train))
print("predictions_test = "+str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()                    #在当前图形上获取与给定关键字args相匹配的当前实例，或创建一个。
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
init_utils.plot_decision_boundary(lambda x:init_utils.predict_dec(parameters,x.T),train_X,train_Y)


#-----------------------------------------------随机初始化-----------------------------------------------
def initialize_parameters_random(layers_dim):
    parameters={}

    np.random.seed(3)
    for i in range(1,len(layers_dim)):
        parameters["W"+str(i)]=np.random.randn(layers_dim[i],layers_dim[i-1])*10
        parameters["b"+str(i)]=np.zeros((layers_dim[i],1))

        assert parameters["W"+str(i)].shape == (layers_dim[i],layers_dim[i-1])
        assert parameters["b"+str(i)].shape == (layers_dim[i],1)

    return parameters

#看看实际运行是怎样
'''
我们可以看到误差开始很高。

这是因为由于具有较大的随机权重，最后一个激活(sigmoid)输出的结果非常接近于0或1，
而当它出现错误时，它会导致非常高的损失(inf)。初始化参数如果没有很好地话会导致梯度消失、爆炸，这也会减慢优化算法。
如果我们对这个网络进行更长时间的训练，我们将看到更好的结果，但是使用过大的随机数初始化会减慢优化的速度。
'''
parameters = model(train_X,train_Y,initialization="random",is_plot=True)

print("训练集：")
predictions_train=init_utils.predict(train_X,train_Y,parameters)
print("测试集：")
predictions_test=init_utils.predict(test_X,test_Y,parameters)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)


#----------------------------------------------抑制梯度异常初始化----------------------------------------------
'''
初始化的模型将蓝色和红色的点在少量的迭代中很好地分离出来，总结一下：
1.不同的初始化方法可能导致性能最终不同
2.随机初始化有助于打破对称，使得不同隐藏层的单元可以学习到不同的参数。
3.初始化时，初始值不宜过大。
4,He初始化搭配ReLU激活函数常常可以得到不错的效果。
'''
def initialize_parameters_he(layers_dim):
    parameters={}

    np.random.seed(3)

    for i in range(1,len(layers_dim)):
        parameters["W"+str(i)] = np.random.randn(layers_dim[i],layers_dim[i-1]) * np.sqrt(2/layers_dim[i-1])
        parameters["b"+str(i)] = np.zeros((layers_dim[i],1))

        assert parameters["W"+str(i)].shape == (layers_dim[i],layers_dim[i-1])
        assert parameters["b"+str(i)].shape == (layers_dim[i],1)

    return parameters

parameters = model(train_X,train_Y,initialization="he",is_plot=True)
print("训练集：")
predictios_train=init_utils.predict(train_X,train_Y,parameters)
print("测试集：")
predictios_test=init_utils.predict(train_X,train_Y,parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5, 1.5])
axes.set_ylim([-1.5, 1.5])
init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), train_X, train_Y)
