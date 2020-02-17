import numpy as np
import h5py

def load_dataset():
    #读取训练集
    train_file=h5py.File('datasets/train_catvnoncat.h5','r')
    train_x_orig=np.array(train_file["train_set_x"][:])
    train_y_orig=np.array(train_file["train_set_y"][:])

    #读取测试集
    test_file=h5py.File('datasets/test_catvnoncat.h5','r')
    test_x_orig=np.array(test_file["test_set_x"][:])
    test_y_orig=np.array(test_file["test_set_y"][:])

    classes=np.array(test_file['list_classes'][:])

    #将y向量变成一行多列的向量，真实值
    train_y_orig=train_y_orig.reshape((1,train_y_orig.shape[0]))
    test_y_orig=test_y_orig.reshape((1,test_y_orig.shape[0]))

    return train_x_orig,train_y_orig,test_x_orig,test_y_orig,classes