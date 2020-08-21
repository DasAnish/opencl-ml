from layer import *
from clobject import *
from neuralnet import NeuralNet


xys = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1)
]
s = []
for x, y in xys:
    x = np.array(x).astype(np.float32)
    y = pycl_array.to_device(
        ClSingleton.get_instance().queue,
        np.array([y]).astype(np.float32)
    )
    s.append((x, y))
xys = s
nn = NeuralNet([
    Layer(2),
    Activation(16),
    Layer(16),
    Activation(1),
    Output(1)
])
for x, y in s:
    print(x, nn.predict(x), y)
nn.train(s, batch_size=4, num_epochs=300)
print()
for x, y in s:
    print(x, nn.predict(x), y)