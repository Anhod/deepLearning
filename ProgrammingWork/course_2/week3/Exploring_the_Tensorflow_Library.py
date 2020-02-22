import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset,random_mini_batches,convert_to_one_hot,predict

np.random.seed(1)

#-----------------------------------------------------------------------------------------------------------------------
'''
    Now that you have imported the library, we will walk you through its different applications. 
You will start with an example, where we compute for you the loss of one training example.
        loss=L(y^,y)=(y^(i)−y(i))2    (对应相减的平方)
'''
y_hat = tf.constant(36,name='y_hat')        ## Define y_hat constant. Set to 36.
y = tf.constant(39,name='y')                ## Define y. Set to 39

loss = tf.Variable((y - y_hat)**2,name='loss')      ## Create a variable for the loss

init = tf.global_variables_initializer()            # When init is run later (session.run(init)),
                                                    # the loss variable will be initialized and ready to be computed
with tf.Session() as session:                       # Create a session and print the output
    session.run(init)                               # Initializes the variables
    print(session.run(loss))                        # Prints the loss

'''
在TensorFlow中编写和运行程序的步骤如下:
    1.创建尚未执行/计算的张量(变量)。
    2.写出这些张量之间的运算。
    3.初始化你的张量。
    4.创建一个会话。
    5.运行会话。这将运行您在上面编写的操作。

因此，当我们为损失创建一个变量时，我们简单地将损失定义为其他数量的函数，但不计算它的值。
为了求值，我们必须运行init=tf.global_variables_initializer()。在最后一行中，我们终于能够计算loss的值并打印它的值。
'''

#-----------------------------------------------------------------------------------------------------------------------
a = tf.constant(2)
b = tf.constant(10)
c = tf.multiply(a,b)
print(c)
'''
    As expected, you will not see 20! You got a tensor(张量) saying that the result is a tensor that does not have the shape attribute, 
and is of type "int32". All you did was put in the 'computation graph', but you have not run this computation yet. 
In order to actually multiply the two numbers, you will have to create a session(对话) and run it.
'''

sess = tf.Session()
print(sess.run(c))
#Great! To summarize, remember to initialize your variables, create a session and run the operations inside the session.

#-----------------------------------------------------------------------------------------------------------------------
'''
    Next, you'll also have to know about placeholders(占位符). A placeholder is an object whose value you can specify only later. 
To specify(指定) values for a placeholder, you can pass(传递) in values by using a "feed dictionary" (`feed_dict` variable). 
    Below, we created a placeholder for x. This allows us to pass in a number later when we run the session. 
'''
x = tf.placeholder(tf.int64,name = 'x')
print(sess.run(2 * x, feed_dict = {x:3}))
sess.close()
'''
    When you first defined x you did not have to specify a value for it. A placeholder is simply a variable that you will 
assign data to only later, when running the session. We say that you feed data to these placeholders when running the session.

    Here's what's happening: When you specify the operations needed for a computation, you are telling TensorFlow how to 
construct a computation graph. The computation graph can have some placeholders whose values you will specify only later. 
Finally, when you run the session, you are telling TensorFlow to execute the computation graph.
'''

#-----------------------------------------------------------------------------------------------------------------------
#Linear function
'''
    Lets start this programming exercise by computing the following equation:  Y = WX + b,where W and X are random matrices 
and b is a random vector.
    Exercise: Compute WX+b where W,X and b are drawn from a random normal distribution. W is of shape (4, 3), X is (3,1) 
and b is (4,1). As an example, here is how you would define a constant X that has shape (3,1):
        X = tf.constant(np.random.randn(3,1), name = "X")
'''
def linear_function():
    np.random.seed(1)

    X = tf.constant(np.random.randn(3,1),name = "X")
    W = tf.constant(np.random.randn(4,3),name = "W")
    b = tf.constant(np.random.randn(4,1),name = "b")
    Y = tf.add(tf.matmul(W,X),b)

    sess = tf.Session()
    result = sess.run(Y)            #run里面的参数是我们想要计算得出的结果

    sess.close()                    #记得关闭会话

    return result

print( "result = \n" + str(linear_function()))