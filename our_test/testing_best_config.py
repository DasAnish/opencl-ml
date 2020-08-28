import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
import time
import threading
from our_test.Timing_decorator import *

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

N = 256 * 16
M = 256 * 16
input_layer = pycl_array.to_device(queue, np.zeros(N).astype(np.float32))
output_layer = pycl_array.to_device(queue, np.zeros(M).astype(np.float32))
matrix = pycl_array.to_device(queue, np.ones(M * N).astype(np.float32))
l = np.random.rand(N).astype(np.float32)

custom_code = '''

#define ROW_DIM 0
#define COL_DIM 1
__kernel void matrix_vector_mul(
    const unsigned int n,
   __global const float *input,
   __global const float *matrix,
   __global float *output)
{

  __local float acc[WPT][TS];
  for (int row=0; row<WPT; row++)
    for(int col=0; col<TS; col++)
      acc[row][col] = 0.0f;

  const int loc_col = get_local_id(COL_DIM); // Max is TS
  const int loc_row = get_local_id(ROW_DIM); // max is WPT
  const int glb_row = get_global_id(ROW_DIM);
  output[glb_row] = 0.0f;

//  printf("row: %d, col: %d, global_row: %d", loc_row, loc_col, glb_row);

  for (int it = 0; it < n/TS+1; it++) {

    const int tiledIter = TS*it + loc_col;
    if (tiledIter < n) {
//      printf("I[tI]: %f|   tI: %d|   M[index]: %f|   index: %d|   input_size: %d",
//       input[tiledIter], tiledIter, matrix[glb_row*n + tiledIter], glb_row*n + tiledIter, n);
//      barrier(CLK_LOCAL_MEM_FENCE);
      acc[loc_row][loc_col] +=  input[tiledIter] * matrix[glb_row*n + tiledIter];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

  }

  int cols = get_local_size(COL_DIM);
  while (cols > 1) {

    cols >>= 1;
    if (loc_col < cols) acc[loc_row][loc_col] += acc[loc_row][loc_col + cols];
    barrier(CLK_LOCAL_MEM_FENCE);

  }

  if (loc_col == 0) output[glb_row] += acc[loc_row][0];
}
'''

configs = '''
#define TS %(ts)d
#define WPT %(wpt)d
'''

ts = 8
programs = {}
while ts <= 128:
    wpt = 256 // ts
    config = configs % {'ts': ts, 'wpt': wpt}
    custom = config + custom_code
    programs[ts] = cl.Program(context, custom).build()

    ts <<= 1


@time_this_function
def func(program, TS, inp_layer, mat, out_layer):
    program.matrix_vector_mul(
        queue,
        (M, TS),
        (256//TS, TS),
        np.int32(N),
        inp_layer.data,
        mat.data,
        out_layer.data
    )


times = {t: 0.0001 for t in programs}
stores = {t: 0 for t in programs}
for i in range(1, 100):
    M = N = np.int32(256)
    input_layer = pycl_array.to_device(queue, np.zeros(N).astype(np.float32))
    output_layer = pycl_array.to_device(queue, np.zeros(M).astype(np.float32))
    matrix = pycl_array.to_device(queue, np.ones(M * N).astype(np.float32))
    for _ in range(250):
        l = np.random.rand(N).astype(np.float32)
        for TS in programs:
            program = programs[TS]

            times[TS] += func(program, TS, input_layer, matrix, output_layer)
    print(f"*************************{i}*********************************")
    print({t: '%0.5f'%(times[t]) for t in times})
    stores = {t: stores[t] + times[t] for t in times}
    times = {t: 0.0001 for t in times}
print(stores)

