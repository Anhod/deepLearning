import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from lr_utils import load_dataset

train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes=load_dataset()
lena=mpimg.imread(train_set_x_orig[25])
print(lena.shape)