import time
from neuralnet import NeuralNet
from layer import *
# from Activations import *

import pyopencl as cl
import numpy as np

import os
os.chdir('..')

I = 4
O = 1

# activation = Activation(I, activation_type=SIGMOID)
# print("Activation: ", activation.layer)
# output = Output(I)
# print("Output: ", output.layer)
#
# activation.set_next_layer(output.layer)
# clS = ClSingleton.get_instance()
# code = Code.get_instance()
# code.program.activate(
#     clS.queue,
#     activation.layer.shape,
#     None,
#     activation.layer.data,
#     output.layer.data,
#     activation.activation_type
# )
# print("Activation: ", activation.layer)
# print("Output: ", output.layer)

nn = NeuralNet(layers=[
    Layer(I),
    Activation(I),
    Layer(I),
    Activation(O),
    Output(O)
])

print(nn)
nn.forward(np.ones(I).astype(np.float32))
print(nn)
nn.forward(np.zeros(I).astype(np.float32))
print(nn)


# l = Layer(queue, context, 4, 16)
# print(l.input_vec)
# print(l.output_vec)
# l.forward()
# print(l.output_vec)
# L3 = L2 = 10000
# L1=100
# nn = NeuralNet(layers=[
#     Layer(L1),
#     Layer(L2),
#     Layer(L3),
#     Layer(L1)
# ])
#
# # nn.layers[0].weights.set(np.ones(16*4).astype(np.float32))
# # nn.layers[0].bias.set(np.zeros(16).astype(np.float32))
# # nn.layers[1].weights.set(np.ones(16*2).astype(np.float32))
# # nn.layers[1].bias.set(np.zeros(2).astype(np.float32))
# # print(nn)
# # nn.layers[0].forward(np.array([1, 1, 1, 1]).astype(np.float32))
# # nn.forward(np.random.rand(L1).astype(np.float32))
# # print(nn)
#
#
# from our_test.Timing_decorator import *
#
#
# @time_this_function
# def nn_forward(_input):
#     nn.forward(_input)
#     # print(nn)
#
#
# @time_this_function
# def np_forward(_input):
#     nn.layers[0].layer.set(_input)
#     for layer in nn.layers[:-1]:
#         vec = layer.layer.get()
#         # print("printing VEC***",vec)
#         bias = layer.bias.get()
#         mat = layer.weights.get().reshape(len(bias), len(vec))
#         next_layer = np.add(np.matmul(mat, vec), bias)
#         layer.next_layer.set(next_layer)
#
#
# snn=0
# snp=0
# A=10
# # ones = pycl_array.to_device(nn.cl.queue, np.ones(100).astype(np.float32))
# # nn.layers[0].set_weights(ones)
# for _ in range(1, A+1):
#     start = time.time()
#     # nn.layers[0].layer.set(np.random.uniform(-1, 1, L1).astype(np.float32))
#     # nn.forward()
#     # # for layer in nn.layers[:-1]:
#     # #     layer.code.program.forward(
#     # #         nn.cl.queue,
#     # #         layer.next_layer.shape,
#     # #         None,
#     # #         layer.layer_size,
#     # #         layer.layer.data,
#     # #         layer.weights.data,
#     # #         layer.bias.data,
#     # #         layer.next_layer.data,
#     # #         np.int32(layer.activation)
#     # #     )
#     # end=time.time()
#     # s+= end-start
#     # print(s, 's/', _)
#
#     _input = np.ones(L1).astype(np.float32)
#
#     snn += nn_forward(_input)
#     print(f"NN took {snn}/{_} | ", end="")
#     # print(nn.layers[0].layer)
#     # print(nn.layers[0].weights)
#     # print(nn.layers[1].layer)
#
#     # output_nn = nn.layers[-1].layer.get()
#
#     snp += np_forward(_input)
#     print(f"NP took {snp}/{_}")
#
#     # output_np = nn.layers[-1].layer.get()
#
#     # print([output_nn[i] == output_np[i] for i in range(L1)])
#
#
#     # print(output_nn)
#     # print(output_np)
#     # break
#
# print(f'ratio {snp/snn}')
#
# # print(f'Average = {snn} / {A} {snn/A}s')