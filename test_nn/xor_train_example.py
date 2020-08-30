from layer import *
from clobject import *
from neuralnet import NeuralNet


xys = []
nums = [0, 1, 3, 2, 6, 14, 10, 11, 9, 8, 12, 4, 5, 7, 15, 13]
num2 = [0, 1, 5, 4, 12, 8, 9, 13, 15, 11, 10, 14, 6, 7, 3, 2]
num3 = [0, 4, 12, 8, 9, 13, 5, 1, 3, 7, 15, 11, 10, 14, 6, 2]
num4 = [0, 8, 10, 2, 3, 7, 6, 4, 5, 1, 9, 13, 12, 14, 15, 11]
num5 = [i for i in range(16)]
np.random.shuffle(num5)
for i in range(16):
    x = np.binary_repr(i, 4)
    x = [int(j) for j in x]
    y = np.binary_repr(num4[i], 4)
    y = [int(j) for j in y]
    xys.append((
        np.array(x).astype(np.float32),
        np.array(y).astype(np.float32)
    ))

nn = NeuralNet(
    Layer(4, activation_type=TANH),
    Layer(64, activation_type=SIGMOID),
    Output(4)
)


nn.predict_values(xys)
nn.train(
    xys,
    batch_size=len(xys),
    num_epochs=4500,
    print_every=100,
    predict_after_every_batch=True
)


print("**************FINAL PREDICTION***************************")
nn.predict_values(xys)
