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
plt.show()
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

print("number of training example = " + str(X_train.shape[1]))
print("number of test example = " + str(X_test.shape[1]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))

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

X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))


#----------------------------2.Initializing the parameters
def initialize_parameters():
    '''
        Initializes parameters to build a neural network with tensorflow. The shapes are:
                W1 : [25, 12288]
                b1 : [25, 1]
                W2 : [12, 25]
                b2 : [12, 1]
                W3 : [6, 12]
                b3 : [6, 1]
    '''
    tf.set_random_seed(1)

    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())

    parameters={"W1":W1,"b1":b1,"W2":W2,"b2":b2,"W3":W3,"b3":b3}
    return parameters

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


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
