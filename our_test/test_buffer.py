import time
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_arr

N = 256*8
context = cl.create_some_context()
queue = cl.CommandQueue(context)
mf = cl.mem_flags

# input_layer = np.random.uniform(-10, 10, size=N).astype(np.float32)

input_layer = cl_arr.to_device(
    queue,
    np.random.rand(N).astype(np.float32)
)

# input_buffer = cl.Buffer(context,
#                          mf.READ_WRITE ,
#                          input_layer.nbytes)

program = cl.Program(context,
"""
__kernel void prg ( __global float *out, __global const float *in) {
  int i = get_global_id(0);
  out[i] += in[i];
}""").build()
# start = time.time()
# program.prg(
#     queue,
#     input_layer.shape,
#     None,
#     input_layer.data
# )
# end = time.time()
# cl.enqueue_copy(queue, input_layer, input_buffer)

# for i in input_layer: print(i, end=", ")

for i in range(1, 64):
    N = i*256
    out = cl_arr.to_device(
        queue,
        np.random.rand(N).astype(np.float32)
    )
    _in = cl_arr.to_device(
        queue,
        np.random.rand(N).astype(np.float32)
    )
    start = time.time()
    for _ in range(1000):
        program.prg(
            queue,
            out.shape,
            None,
            out.data,
            _in.data
        )
    end=time.time()
    print(f"{i}*256 takes: {end-start}s")