from layer import *
from neuralnet import NeuralNet
from clobject import *

cl = ClSingleton.get_instance()
code = Code.get_instance()

I = 15
O = 15

nn = NeuralNet(Layer(I, SIGMOID), Output(O))

print(nn)
nn.layers[0].weights.set(
    np.array([[i*j/100 for i in range(I)] for j in range(O)]).astype(np.float32)
)
print(nn.layers[0].weights)
print("*********************************************************************")
nn.forward(np.ones(I).astype(np.float32))
print(nn)

nn.output_layer.expected.set(np.ones(O).astype(np.float32))

## testing back

error_vec = nn.output_layer.get_error_derivative()
print(error_vec)