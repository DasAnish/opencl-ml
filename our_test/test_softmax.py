from clobject import *
import numpy as np

cl = ClSingleton.get_instance()
code = Code.get_instance()


layer = np.random.random(10).astype(np.float32)
layer_cl = pycl_array.to_device(
    cl.queue,
    layer
)

acc = pycl_array.to_device(
    cl.queue, np.zeros(0).astype(np.float32)
)

code.program.activate(
    cl.queue,
    layer_cl.shape,
    None,
    layer_cl.data,
    np.int32(6),
    acc.data
)

print(layer_cl)

layer = np.exp(layer)
layer /= np.sum(layer)
print(layer)

print(layer == layer_cl.get())
