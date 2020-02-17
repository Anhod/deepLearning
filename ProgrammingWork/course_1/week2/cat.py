import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()

'''
第一部分：
    查看第26张图片是什么
'''
index=25
plt.imshow(train_set_x_orig[index])                #显示图像
plt.show()

'''
第二部分：
    结合一下训练集里面的数据来看一下我到底都加载了一些什么东西。
    训练标签值   
    使用np.squeeze的目的是压缩维度，【未压缩】train_set_y[:,index]的值为[1] , 【压缩后】np.squeeze(train_set_y[:,index])的值为1
    只有压缩后的值才能进行解码工作

    # print(train_set_y_orig.shape)
    # print(train_set_y_orig)
    # print(np.squeeze(train_set_y_orig).shape)
    # print(np.squeeze(train_set_y_orig))
    # print(np.squeeze(train_set_y_orig[:,index]))
'''
print("y="+str(train_set_y_orig[:,index])+",it's a "+classes[np.squeeze(train_set_y_orig[:,index])].decode("utf-8")+"' picture")

'''
第三部分：
    查看数据集中的基本数据
'''
m_train=train_set_y_orig.shape[1]                 #训练集中的图片数量
m_test=test_set_y_orig.shape[1]                   #测试集中的图片数量
num_px=train_set_x_orig.shape[1]                  #每张图片的长/宽

print("训练集中的图片数量："+str(m_train))
print("测试集中的图片数量："+str(m_test))
print("每张图片的长/宽："+str(num_px))
print("每张图片的大小："+str(num_px)+","+str(num_px)+",3")
print("训练集中的图片维数："+str(train_set_x_orig.shape))
print("训练集标签的维数："+str(train_set_y_orig.shape))
print("测试集中的图片维数："+str(test_set_x_orig.shape))
print("测试及标签的维数："+str(test_set_y_orig.shape))

'''
第四部分：
    为了方便，我们要把维度为（64，64，3）的numpy数组重新构造为（64 x 64 x 3，1）的数组，
    要乘以3的原因是每张图片是由64x64像素构成的，而每个像素点由（R，G，B）三原色构成的，所以要乘以3。
    在此之后，我们的训练和测试数据集是一个numpy数组，【每列代表一个平坦的图像】 ，应该有m_train和m_test列,64 x 63 x 3行
    可以这样做(记得转置)：
'''
#这时候的矩阵就成了X矩阵，先把矩阵的列数确定，也就是一幅图片就是一列，传入参数-1让程序自动计算行数（记得转置）
train_set_x_flatten=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

'''
第五部分：
    为了表示彩色图像，必须为每个像素指定红色，绿色和蓝色通道（RGB），因此像素值实际上是从0到255范围内的三个数字的向量。
    机器学习中一个常见的预处理步骤是对数据集进行居中和标准化，这意味着可以减去每个示例中整个numpy数组的平均值，然后将每个示例除以整个numpy数组的标准偏差。
    但对于图片数据集，它更简单，更方便，几乎可以将数据集的每一行除以255（像素通道的最大值），因为在RGB中不存在比255大的数据，所以我们可以放心的除以255，
    让标准化的数据位于[0,1]之间，现在标准化我们的数据集：
'''
train_set_x=train_set_x_flatten/255
test_set_x=test_set_x_flatten/255

'''
建立神经网络的主要步骤是：
    1.定义模型结构（例如输入特征的数量）
    2.初始化模型的参数
    3.循环：
        3.1 计算当前损失（正向传播）
        3.2 计算当前梯度（反向传播）
        3.3 更新参数（梯度下降）
'''

#定义sigmoid()函数
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
print("=========测试sigmoid函数=========")
print("sigmoid(0)=",sigmoid(0))
print("sigmoid(9.2)=",sigmoid(9.2))

#初始化我们想要的参数w和b
def initialize_with_zeros(dim):
    '''
    创建一个维度为(dim,1)的零向量以及把b初始化为0
    :param dim:我们想要的w矢量的大小
    :return:
        w：维度为(dim,1)的零矢量
        b：初始化为0
    '''
    w=np.zeros(shape=(dim,1))
    b=0

    #使用断言来确保我要的数据是正确的
    assert (w.shape==(dim,1))
    assert (isinstance(b,float) or isinstance(b,int))

    return (w,b)

#执行"前向"和"后向"传播步骤来学习参数，对传入的数据集进行整体迭代一次
def propagate(w,b,X,Y):
    '''
        实现前向和后向传播的成本函数及其梯度。
        :param w: 权重，大小不等的数组（num_px * num_px * 3 , 1）
        :param b:偏差，一个标量
        :param X: 矩阵类型为（num_px * num_px * 3,训练数量）
        :param Y: 真正的"标签"矢量，（如果非猫则为0，是猫则为1），矩阵维度为（1，训练数据数量）
        :return:
    '''

    m = X.shape[1]                        #图片数量

    #正向传播
    z = np.dot(w.T,X) + b
    a = sigmoid(z)                        #计算激活值
    cost = (-1 / m) * np.sum(Y * np.log(a) + (1 - Y) * (np.log(1 - a)))     #计算成本,相当于J,cost function

    #反向传播,偏导公式
    dz = a - Y
    dw = ( 1 / m ) * np.dot( X , dz.T )           #w初始为向量，所以不用求和
    db = ( 1 / m ) * np.sum( dz )

    #使用断言来确保数据是正确的
    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    #创建一个字典，把dw和db保存起来
    grads = {
        "dw":dw,
        "db":db
    }
    return (grads,cost)
#测试一下propagate
print("========测试proparate========")
w,b,X,Y=np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
grads,cost=propagate(w,b,X,Y)
print("dw="+str(grads["dw"]))
print("db="+str(grads["db"]))
print("cost="+str(cost))

'''
现在，我要使用渐变下降更新参数。
目标是通过最小化成本函数 J来学习 w和b 。对于参数 θ\thetaθ ，更新规则是 $ \theta = \theta - \alpha \text{ } d\theta$，其中 α\alphaα 是学习率。
'''
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    此函数通过运行梯度下降算法来优化w和b

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数组。
        Y  - 真正的“标签”矢量（如果非猫则为0，如果是猫则为1），矩阵维度为(1,训练数据的数量)
        num_iterations  - 优化循环的迭代次数
        learning_rate  - 梯度下降更新规则的学习率
        print_cost  - 每100步打印一次损失值

    返回：
        params  - 包含权重w和偏差b的字典
        grads  - 包含权重和偏差相对于成本函数的梯度的字典
        成本 - 优化期间计算的所有成本列表，将用于绘制学习曲线。

    提示：
    我们需要写下两个步骤并遍历它们：
        1）计算当前参数的成本和梯度，使用propagate（）。
        2）使用w和b的梯度下降法则更新参数。
    """

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # 梯度下降
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))

    params = {
        "w": w,
        "b": b}
    grads = {
        "dw": dw,
        "db": db}
    return (params, grads, costs)
#测试优化数据
print("========测试optimize========")
w,b,X,Y=np.array([[1],[2]]),2,np.array([[1,2],[3,4]]),np.array([[1,0]])
params,grads,costs=optimize(w,b,X,Y,num_iterations=100,learning_rate=0.009,print_cost=False)
print("w = "+str(params["w"]))
print("b = "+str(params["b"]))
print("dw = "+str(grads["dw"]))
print("db = "+str(grads["db"]))

'''
optimize函数会输出已学习的w和b的值，我们可以使用w和b来预测数据集X的标签。
现在我们要实现预测函数predict（）。
'''
def predict(w, b, X):
    """
    使用学习逻辑回归参数logistic （w，b）预测标签是0还是1，

    参数：
        w  - 权重，大小不等的数组（num_px * num_px * 3，1）
        b  - 偏差，一个标量
        X  - 维度为（num_px * num_px * 3，训练数据的数量）的数据

    返回：
        Y_prediction  - 包含X中所有图片的所有预测【0 | 1】的一个numpy数组（向量）
    """

    m = X.shape[1]          # 图片的数量
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)    #和传入的数据集的数据特征行数相同

    # 计预测猫在图片中出现的概率,yhat
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        # 将概率a [0，i]转换为实际预测p [0，i]
        Y_prediction[0, i] = 1 if A[0, i] > 0.5 else 0

    # 使用断言
    assert (Y_prediction.shape == (1, m))
    return Y_prediction
#测试predict
print("====================测试predict====================")
w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3,4]]), np.array([[1, 0]])
print("predictions = " + str(predict(w, b, X)))

'''
就目前而言，我们基本上把所有的东西都做完了，现在我们要把这些函数统统整合到一个model()函数中，
届时只需要调用一个model()就基本上完成所有的事了。
'''


def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    '''

    :param X_train: numpy的数组,维度为（num_px * num_px * 3，m_train）的训练集
    :param Y_train: numpy的数组,维度为（1，m_train）（矢量）的训练标签集
    :param X_test: numpy的数组,维度为（num_px * num_px * 3，m_test）的测试集
    :param Y_test: numpy的数组,维度为（1，m_test）的（向量）的测试标签集
    :param num_iterations: 表示用于优化参数的迭代次数的超参数
    :param learning_rate: 表示optimize（）更新规则中使用的学习速率的超参数
    :param print_cost: 设置为true以每100次迭代打印成本
    :return:
        d  - 包含有关模型信息的字典。
    '''
    w, b = initialize_with_zeros(X_train.shape[0])               #初始化为列向量

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    # 从字典“参数”中检索参数w和b
    w, b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子，预测的是yhat
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性,因为两个预测集合中的值非0即1，相差的结果是1的话则预测错误，平均值是错误率
    print("训练集准确性：", format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100), "%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d
print("====================测试model====================")
#这里加载的是真实的数据，请参见上面的代码部分。
d = model(train_set_x, train_set_y_orig, test_set_x, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

