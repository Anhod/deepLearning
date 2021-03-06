import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset,random_mini_batches,convert_to_one_hot,predict

np.random.seed(1)
'''
In this part of the assignment you will build a neural network using tensorflow. Remember that there are two parts to implement a tensorflow model:
    ·Create the computation graph
    ·Run the graph
'''
#--------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------1.Problem statement:SIGNS Dataset
'''
    One afternoon, with some friends we decided to teach our computers to decipher sign language. We spent a few hours taking pictures in front 
of a white wall and came up with the following dataset. It's now your job to build an algorithm that would facilitate communications from a 
speech-impaired(语言障碍) person to someone who doesn't understand sign language.
        ·Training set: 1080 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (180 pictures per number).
        ·Test set: 120 pictures (64 by 64 pixels) of signs representing numbers from 0 to 5 (20 pictures per number).
        
    参见hands.png，这是未降低图像分辨率的时候
'''
X_train_org,Y_train_org,X_test_org,Y_test_org,classes = load_dataset()

#查看图片
index = 22
plt.imshow(X_train_org[index])
#plt.show()
print("y = " + str(np.squeeze(Y_train_org[:,index])))
print(X_train_org.shape)

#对数据进行处理
X_train_flatten = X_train_org.reshape(X_train_org.shape[0],-1).T
X_test_flatten = X_test_org.reshape(X_test_org.shape[0],-1).T

#归一化
X_train = X_train_flatten / 255
X_test = X_test_flatten / 255

#将Y标签转为one_hot_matrix
Y_train = convert_to_one_hot(Y_train_org,6)
Y_test = convert_to_one_hot(Y_test_org,6)

# print("number of training example = " + str(X_train.shape[1]))
# print("number of test example = " + str(X_test.shape[1]))
# print("X_train shape: " + str(X_train.shape))
# print("Y_train shape: " + str(Y_train.shape))
# print(X_train.dtype)
# print("X_test shape: " + str(X_test.shape))
# print("Y_test shape: " + str(Y_test.shape))

'''
    Your goal is to build an algorithm capable of recognizing a sign with high accuracy. To do so, you are going to build a tensorflow model 
that is almost the same as one you have previously built in numpy for cat recognition (but now using a softmax output). It is a great occasion 
to compare your numpy implementation to the tensorflow one.

    The model is LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX. The SIGMOID output layer has been converted to a SOFTMAX. A SOFTMAX 
layer generalizes SIGMOID to when there are more than two classes.
'''


#----------------------------1.Create placeholders
'''
    Your first task is to create placeholders for X and Y. This will allow you to later pass your training data in when you run your session.
'''
def create_placeholders(n_x,n_y):
    '''
    Creates the placeholders for the tensorflow session.

    :param n_x:scalar(标量), size of an image vector (num_px * num_px = 64 * 64 * 3 = 12288)
    :param n_y:scalar, number of classes (from 0 to 5, so -> 6)
    :return:
         X -- placeholder for the data input, of shape [n_x, None] and dtype "tf.float32"
         Y -- placeholder for the input labels, of shape [n_y, None] and dtype "tf.float32"

     Tips:
        - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
        - In fact, the number of examples during test/train is different.
    '''
    X = tf.placeholder(tf.float32,[n_x,None],name = "X")
    Y = tf.placeholder(tf.float32,[n_y,None],name = "Y")

    return X,Y



# X, Y = create_placeholders(12288, 6)
# print ("X = " + str(X))
# print ("Y = " + str(Y))


#----------------------------2.Initializing the parameters
def initialize_parameters():
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())

    parameters = {"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
    return parameters

# tf.reset_default_graph()
# with tf.Session() as sess:
#     parameters = initialize_parameters()
#     print("W1 = " + str(parameters["W1"]))
#     print("b1 = " + str(parameters["b1"]))
#     print("W2 = " + str(parameters["W2"]))
#     print("b2 = " + str(parameters["b2"]))


#----------------------------3.Forward Propagation in Tensorflow
'''
    You will now implement the forward propagation module in tensorflow. The function will take in a dictionary of parameters and it will 
complete the forward pass. The functions you will be using are:
        -tf.add(...,...) to do an addition
        -tf.matmul(...,...) to do a matrix multiplication
        -tf.nn.relu(...) to apply the ReLU activation
        
    Question: Implement the forward pass of the neural network. We commented for you the numpy equivalents so that you can compare the tensorflow 
implementation to numpy. It is important to note that the forward propagation stops at z3. The reason is that in tensorflow the last linear layer 
output is given as input to the function computing the loss. Therefore, you don't need a3!
'''
def forward_propagation(X,parameters):
    """
       Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX

       Arguments:
       X -- input dataset placeholder, of shape (input size, number of examples)
       parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                     the shapes are given in initialize_parameters

       Returns:
       Z3 -- the output of the last LINEAR unit
       """
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add( tf.matmul(W1,X) , b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add( tf.matmul(W2,A1) , b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add( tf.matmul(W3,A2) , b3)

    return Z3

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     print("Z3: " + str(Z3))


#----------------------------4.Compute cost
'''
As seen before, it is very easy to compute the cost using:
        tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = ..., labels = ...))
        
Question: Implement the cost function below.
        ·It is important to know that the "logits" and "labels" inputs of tf.nn.softmax_cross_entropy_with_logits 
    are expected to be of shape (number of examples, num_classes). We have thus transposed Z3 and Y for you.
        ·Besides, tf.reduce_mean basically does the summation over the examples.
'''
def compute_cost(Z3,y):
    """
        Computes the cost

        Arguments:
        Z3 -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
        Y -- "true" labels vector placeholder, same shape as Z3

        Returns:
        cost - Tensor of the cost function
        """
    logits = tf.transpose(Z3)
    labels = tf.transpose(y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))

    return cost

# tf.reset_default_graph()
# with tf.Session() as sess:
#     X,Y = create_placeholders(12288,6)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X,parameters)
#     cost = compute_cost(Z3,Y)
#     print("cost = " + str(cost))


#----------------------------5.Backward propagation & parameters updates
'''
    This is where you become grateful to programming frameworks. All the backpropagation and the parameters update is taken care of in 1 line of 
code. It is very easy to incorporate this line in the model.

    After you compute the cost function. You will create an "optimizer" object. You have to call this object along with the cost when running 
the tf.session. When called, it will perform an optimization on the given cost with the chosen method and learning rate.
    For instance, for gradient descent the optimizer would be:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
    To make the optimization you would do:
        _ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
    This computes the backpropagation by passing through the tensorflow graph in the reverse order. From cost to inputs.
    When coding, we often use _ as a "throwaway" variable to store values that we won't need to use later. Here, _ takes on the evaluated value 
of optimizer, which we don't need (and c takes the value of the cost variable).
    (在编码时，我们经常使用_作为“一次性”变量来存储我们以后不需要的值。这里，_接受优化器的评估值，这是我们不需要的(而c接受成本变量的值)。)
'''


#----------------------------6.Build the model
def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,num_epochs = 1500,minibatch_size = 32,print_cost = True):
    """
        Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

        Arguments:
            X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
            Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
            X_test -- training set, of shape (input size = 12288, number of training examples = 120)
            Y_test -- test set, of shape (output size = 6, number of test examples = 120)
            learning_rate -- learning rate of the optimization
            num_epochs -- number of epochs of the optimization loop
            minibatch_size -- size of a minibatch
            print_cost -- True to print the cost every 100 epochs

        Returns:
            parameters -- parameters learnt by the model. They can then be used to predict.
        """

    ops.reset_default_graph()       #能够在不覆盖tf变量的情况下重新运行模型
    #tf.reset_default_graph()
    tf.set_random_seed(1)           #保持一致的结果
    seed = 3                        #保持一致的结果
    (n_x,m) = X_train.shape         # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]
    costs = []

    #创建占位符
    X,Y = create_placeholders(n_x,n_y)

    #初始化参数
    parameters = initialize_parameters()

    #前向传播
    Z3 = forward_propagation(X,parameters)

    #计算损失
    cost  = compute_cost(Z3,Y)

    #定义优化器
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

    #初始化所有值
    init = tf.global_variables_initializer()

    #创建会话并计算tensorflow graph
    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(num_epochs):
            epoch_cost = 0
            num_minibatches = int( m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)

            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict={ X : minibatch_X, Y : minibatch_Y})
                epoch_cost = epoch_cost + minibatch_cost / minibatch_size

            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        plt.plot(np.squeeze(costs))
        plt.xlabel('cost')
        plt.ylabel('iteration(per five)')
        plt.title("learning_rate = " + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print ("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))


        return parameters

parameters = model(X_train, Y_train, X_test, Y_test)
