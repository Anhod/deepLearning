import numpy as np
import matplotlib.pyplot as plt
import h5py
from lr_util import load_dataset

train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes=load_dataset()

#变成X矩阵
train_set_x=train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x=test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

#标准化数据
train_x=train_set_x/255
test_x=test_set_x/255

#定义sigmoid函数
def sigmoid(z):
    a=1/(1+np.exp(-z))
    return a

#初始化函数
def initialize_with_zeros(dim):
    w = np.zeros(shape=(dim,1))
    b = 0

    assert (w.shape == (dim,1))
    assert (isinstance(b,float) or isinstance(b,int))
    return (w,b)

#对数据集的一次迭代
def propagate(w,b,X,Y):
    m = X.shape[1]        #数据图片数量

    z = np.dot( w.T , X ) + b
    a = sigmoid(z)
    cost = (-1 / m) * np.sum(Y * np.log(a) + (1 - Y) * (np.log(1 - a)))

    dz = a - Y
    dw = ( 1 / m )*np.dot(X , dz.T)
    db = ( 1 / m ) * np.sum( dz )

    assert (dw.shape == w.shape)
    assert (db.dtype == float)

    cost=np.squeeze(cost)
    assert (cost.shape == ())

    grams={
        "dw":dw,
        "db":db
    }
    return (grams,cost)

#优化
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost=False):
    costs=[]

    for i in range(num_iterations):
        grads,cost=propagate(w,b,X,Y)

        dw=grads["dw"]
        db=grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db

        # 记录成本
        if i % 100 == 0:
            costs.append(cost)
        # 打印成本数据
        if (print_cost) and (i % 100 == 0):
            print("迭代的次数: %i ， 误差值： %f" % (i, cost))

    params={
        "w":w,
         "b":b
    }

    grads={
        "dw":dw,
        "db":db
    }
    return (params,grads,costs)

def predict(w,b,X):
    m=X.shape[1]
    y_prediction=np.zeros((1,m))
    w = w.reshape(X.shape[0],1)

    z=np.dot(w.T,X)+b
    A=sigmoid(z)

    for i in range(A.shape[1]):
        y_prediction[0,i]=1 if A[0,i]>0.5 else 0

    assert (y_prediction.shape == (1, m))

    return y_prediction

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w , b = initialize_with_zeros(X_train.shape[0])

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w , b = parameters["w"], parameters["b"]

    # 预测测试/训练集的例子，预测的是yhat
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # 打印训练后的准确性,因为两个预测集合中的值非0即1，相差的结果是1的话则预测错误，平均值是错误率
    print("训练集准确性：",format(100 - np.mean(np.abs(Y_train-Y_prediction_train))),"%")
    print("测试集准确性：", format(100 - np.mean(np.abs(Y_test - Y_prediction_test))), "%")

    d = {
        "costs": costs,
        "Y_prediction_test": Y_prediction_test,
        "Y_prediciton_train": Y_prediction_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iterations}
    return d

d = model(train_x, train_set_y_orig, test_x, test_set_y_orig, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#绘制图
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()