from layer import *
from clobject import *
from neuralnet import NeuralNet


xys = []

for i in range(16):
    x = np.binary_repr(i, 4)
    x = [int(j) for j in x]
    xys.append((
        np.array(x).astype(np.float32),
        np.array(x).astype(np.float32)
    ))

nn = NeuralNet(
    Layer(4, activation_type=RELU),
    Layer(32, activation_type=SIGMOID),
    Layer(128, activation_type=SIGMOID),
    Layer(128, activation_type=SIGMOID),
    Layer(32, activation_type=SIGMOID),
    Output(4)
)


nn.predict_values(xys)
nn.train(xys, batch_size=len(xys)*2, num_epochs=5000, print_every=100)
nn.predict_values(xys)
print(nn)