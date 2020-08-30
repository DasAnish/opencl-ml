import os
import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np
import matplotlib.pyplot as plt
from layer import *
from neuralnet import NeuralNet
from clobject import *

os.chdir('..')
cls = ClSingleton.get_instance()

image_size = 28

no_of_different_labels=10
image_pixels = image_size**2

data_path = os.path.join('..', 'data', 'mnist')

train_data = np.loadtxt(os.path.join(data_path, 'mnist_train.csv'),
                        delimiter=',')
print('loaded train values')
test_data = np.loadtxt(os.path.join(data_path, 'mnist_test.csv'),
                        delimiter=',')
print('loaded test values')

len_train_data = len(train_data)
len_test_data = len(test_data)

train_imgs = (np.asfarray(train_data[:, 1:])*0.99 + 0.01).astype(np.float32)
test_imgs = (np.asfarray(test_data[:, 1:])*0.99 + 0.01).astype(np.float32)

label = np.arange(no_of_different_labels)
train_labels = np.asfarray(train_data[:, :1])
test_labels = np.asfarray(test_data[:, :1])
print('train and test images converted to float (0, 1)')

train_labels_one_hot = (label == train_labels).astype(np.float32)
test_labels_one_hot = (label == test_labels).astype(np.float32)
print("train and test one-hot vectors")

# train_labels = np.vectorize(lambda y: pycl_array.to_device(cls.queue,
#                                                            train_labels_one_hot[y]))(np.arange(len_train_data))
# test_labels = np.vectorize(lambda y: pycl_array.to_device(cls.queue,
#                                                           test_labels_one_hot[y]))(np.arange(len_test_data))
# print("train and test labels created")

# train_dataset = np.vectorize(lambda i: (train_imgs[i], train_labels[i]))(np.arange(len_train_data))
# test_dataset = np.vectorize(lambda i: (test_imgs[i], test_labels[i]))(np.arange(len_test_data))
# print("train and test dataset created")

print("Building NN")
nn= NeuralNet(
    Layer(image_pixels, activation_type=TANH),
    # Layer(256, activation_type=SIGMOID),
    # Layer(512, activation_type=SIGMOID),
    # Layer(256, activation_type=SIGMOID),
    # Layer(256, activation_type=SIGMOID),
    # Layer(128, activation_type=SIGMOID),
    # Layer(128, activation_type=SIGMOID),
    # Layer(64, activation_type=SIGMOID),
    # Layer(64, activation_type=SIGMOID),
    Layer(32, activation_type=SOFTMAX),
    Output(no_of_different_labels)
)

batch = 100
nn.fit(train_imgs,
       train_labels_one_hot,
       batch_size=batch,
       num_epochs=len_train_data//batch,
       len_dataset = len_train_data,
       print_every=10)

