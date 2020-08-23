import os
from clobject import *
import numpy as np

cl = ClSingleton.get_instance()
code = Code.get_instance()

len_del = np.int32(9)
len_z = np.int32(11)

_del = pycl_array.to_device(
    cl.queue,
    np.arange(len_del).astype(np.float32)
)

z_val = pycl_array.to_device(
    cl.queue,
    np.arange(len_z).astype(np.float32)
)

weights_del = pycl_array.to_device(
    cl.queue,
    np.zeros(len_del*len_z).astype(np.float32)
)

code.program.weights_del(
    cl.queue,
    (len_z, len_del),
    (16, 16),
    len_z,
    _del.data,
    z_val.data,
    weights_del.data
)

print(weights_del.get().reshape((len_z, len_del)).astype(np.int32))