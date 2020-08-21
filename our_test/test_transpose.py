from our_test.Timing_decorator import *
import pyopencl as cl
import pyopencl.array as cl_array
import numpy  as np


context = cl.create_some_context()
queue = cl.CommandQueue(context)

H=np.int32(5000)
W=np.int32(5000)

matrix = cl_array.to_device(
    queue,
    np.random.rand(H*W).astype(np.float32)
)
transposed = cl_array.zeros_like(matrix)

with open('..\\kernel.cl', 'r') as f:
    code = f.read()

program = cl.Program(context, code).build()

matrix_np = matrix.get().reshape(W, H)
transposed_np = np.transpose(matrix_np).reshape(W*H, 1)


@time_this_function
def opencl():
    program.transpose(
        queue,
        (W, H),
        (16, 16),
        matrix.data,
        transposed.data,
        W,
        H
    )


@time_this_function
def numpy():
    transposed_np = np.transpose(matrix_np)


if __name__ == '__main__':

    snn = 0
    snp = 0

    for i in range(200):
        snn += opencl()
        # print(f'{i+1} OPENCL took {snn} | ', end=" ")
        snp += numpy()
        # print(f'{i+1} NUMPY took {snp}')

        print(f'Ratio : {snp/snn}')
