from layer import *
from neuralnet import NeuralNet
from clobject import *

cl = ClSingleton.get_instance()
code = Code.get_instance()

I = 16
O = 16
nn = NeuralNet(layers=[
    Layer(I, SIGMOID),
    Layer(O, SIGMOID),
    Layer(I, SOFTMAX)],
    output_layer=Output(O)
)

nn.output_layer.expected.set(np.random.rand(O).astype(np.float32))
# nn.layers[0].layer.set(np.array([1, 0]).astype(np.float32))
nn.forward(np.random.rand(I).astype(np.float32))
print(nn)
error_vec = nn.output_layer.get_error_derivative()
# print("error_vec", error_vec)
for i in range(-1, -len(nn.layers)-1, -1):
    layer = nn.layers[i]

    if isinstance(layer, Layer):
        layer.weights_del = pycl_array.to_device(
            cl.queue,
            np.zeros(layer.layer_size * layer.next_layer_size).astype(np.float32)
        )
        layer.transposed = pycl_array.to_device(
            cl.queue,
            np.zeros(layer.layer_size * layer.next_layer_size).astype(np.float32)
        )
        code.program.transpose(cl.queue, (layer.layer_size, layer.next_layer_size),
                              (16, 16), layer.weights.data,
                              layer.transposed.data, layer.layer_size,
                              layer.next_layer_size)
        layer.bias_del: pycl_array.Array = pycl_array.to_device(
            cl.queue,
            np.zeros(layer.next_layer_size).astype(np.float32)
        )
        # if i == -2: print(layer)
        error_vec = layer.backward(error_vec)
        # if i == -2: print(layer)
        print("Weights_del", layer.weights_del)
        print("Bias_del", layer.bias_del)
        print("error_vec", -i,  error_vec)

    else:
        error_vec = layer.backward(error_vec)
        # print("error_vec", -i, error_vec)

