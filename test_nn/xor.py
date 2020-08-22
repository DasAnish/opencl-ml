from layer import *
from clobject import *
from neuralnet import NeuralNet


xys = [
    ([0, 0], [0, 0]),
    ([0, 1], [0, 1]),
    ([1, 0], [1, 0]),
    ([1, 1], [1, 1])
]
s = []
for x, y in xys:
    x = np.array(x).astype(np.float32)
    y = np.array(y).astype(np.float32)
    s.append((x, y))
xys = s
nn = NeuralNet([
    Layer(2),
    Activation(16),
    Layer(16),
    Activation(2),
    Output(2)
])
for x, y in s:
    print(x, nn.predict(x), y)
nn.train(s, batch_size=len(xys), num_epochs=300)
print()
for x, y in s:
    print(x, nn.predict(x), y)