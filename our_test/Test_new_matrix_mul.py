import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
import time
import threading

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

N=256*10
M=256*10
input_layer = pycl_array.to_device(queue, np.zeros(N).astype(np.float32))
output_layer = pycl_array.to_device(queue, np.zeros(M).astype(np.float32))
matrix = pycl_array.to_device(queue, np.ones(M*N).astype(np.float32))
l = np.random.rand(N).astype(np.float32)

code = '''

__kernel void naive(
    const unsigned int n,
   __global const float *input,
   __global const float *matrix,
   __global float *output)
{
  const int i = get_global_id(0);
  output[i] = 0;
  
  for (int k=0; k < n; k++) {
    output[i] += input[k] * matrix[i*n + k];
  }  
}

#define S 256

__kernel void ver1(const unsigned int size,
 __global const float *input,
 __global const float *weights,
 __global float *output)
{

  const int glb = get_global_id(0);
  const int loc = get_local_id(0);
  const int grp = get_group_id(0);
  float acc = 0.0f;
  output[glb] = 0.0f;

  __local float vec_buffer[S];
  __local float mat_buffer[S];

  const int iterations = size/S;
  int iter = 0;

  if (iterations) {
    for (; iter < iterations; iter++) {

      const int tiledIter = S*iter + loc;
      vec_buffer[loc] = input[tiledIter];
      mat_buffer[loc] = weights[glb*size + tiledIter];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int k=0; k<S; k++) {
        acc += vec_buffer[k]*mat_buffer[k];
      }

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    output[glb] = acc;

  } else {
    output[glb] = 0.0f;

    for (int k=0; k<size; k++) {
      output[glb] += input[k]*weights[k];
    }
  }
}

#define TS 8
#define WPT 32
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
}'''

custom_code = '''
#define TS %(ts)d
#define WPT %(wpt)d
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

program = cl.Program(context, code).build()

matrix_np = matrix.get().reshape(N, M)

from our_test.Timing_decorator import *

@print_name_and_time
def naive():
    program.naive(
        queue,
        output_layer.shape,
        None,
        np.int32(N),
        input_layer.data,
        matrix.data,
        output_layer.data
    )

@print_name_and_time
def ver1():
    program.ver1(
        queue,
        output_layer.shape,
        None,
        np.int32(N),
        input_layer.data,
        matrix.data,
        output_layer.data
    )

@print_name_and_time
def gemv():
    program.matrix_vector_mul(
        queue,
        (M, 8),
        (32, 8),
        np.int32(N),
        input_layer.data,
        matrix.data,
        output_layer.data
    )

@print_name_and_time
def numpy():
    # print('numpy', end='')
    output = np.matmul(l, matrix_np)


funcs = {naive: 0.00001, ver1: 0.00001, gemv: 0.00001, numpy: 0.00001}

for _ in range(100):

    l = np.random.rand(N).astype(np.float32)
    input_layer.set(l)
    sum_ = np.sum(l)

    for func in funcs:
        funcs[func] += func()
        print('%0.6f'%(funcs[func]), output_layer[0] == sum_)

    # print(f"naive: {funcs[naive]}   \nver1: {funcs[ver1]}   \nnumpy: {funcs[numpy]}   \nfinal: {funcs[matrix_vector_mul]}")
    # print(f" ratio: {funcs[naive] / funcs[matrix_vector_mul]} || {funcs[ver1] / funcs[matrix_vector_mul]}"
    #       f" || {funcs[numpy]/ funcs[matrix_vector_mul]}")
    # print("output layer", output_layer[0], "actual value: ", sum_)
    print("###############################################################################")
    # break
    # print(np.equal(l, output_layer.get()))

'''
[0. 0. 0. 0. 0.]
[9. 1. 1. 0. 0.]
const int iterations = get_global_size(0)/TS;
  if (iterations) {
    for (int iter = 0; iter < iterations; iter++) {


      for (int w=0; w < WPT; w++) {
        const int locIter = w*RTS + loc; // goes from 0 to TS-1
        const int tiledIter = TS*iter + locIter; // goes from 0 to input_dim
        vec_buffer[locIter] = input[tiledIter];
        mat_buffer[locIter][wpt] = weights[(glb*WPT + wpt)*size + tiledIter];
      }


      //const int tiledIter = TS*iter + loc;
      //vec_buffer[loc] = input[tiledIter];
      //mat_buffer[loc] = weights[glb*size + tiledIter];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int k=0; k<TS; k++) {
        for (int w=0; w<WPT; w++) {
          acc[w] += vec_buffer[k]*mat_buffer[k][w];
        }
      }

    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int w=0; w<WPT; w++)
      output[glb*WPT + w] = acc[w];

  } else {
    output[glb] = bias[glb];

    for (int k=0; k<size; k++) {
      output[glb] += input[k]*weights[k];
    }
  }
  
  
######################################################
  
#define TS 256
#define WPT 4

__kernel void forward(
  const unsigned int ld,
  const unsigned int od,
 __global const float *input,
 __global const float *weights,
 __global const float *bias,
 __global float *output){

  //const int glb = get_global_id(0);
  const int col = get_local_id(0); // Max is TS  32
//  const int row = get_local_id(1); // Max is WPT  8
  const int grp = get_group_id(0);
  const int glb = (grp*TS + col)*WPT; // Max is len(output) / WPT

  const int extra = od % WPT;

  float acc[WPT];
  for (int w=0; w<WPT; w++) {
    if (glb + w < od)
      acc[w] = bias[glb+w];
  }

  __local float vec_buffer[TS];
  __local float mat_buffer[TS][WPT];

  const int iterations = ld/TS;

  for (int iter=0; iter<=iterations; iter++) { // Runs for iterations+1
    const int tiledIter = TS*iter + col;
    if (tiledIter < ld) { // making sure we don't run off stay withing the size of input
      // Loading input into buffer
      vec_buffer[col] = input[tiledIter];

      // Loading values of the
      for(int w=0; w<WPT; w++) {
        // All threads read in 4 values
        if (glb + w < od)
          mat_buffer[col][w] = weights[(glb + w)*ld + tiledIter];
        if (output[glb+w]==0)
          output[glb+w] = tiledIter;
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      // Doing the mathematics
      for (int k=0; k<TS; k++) {
        for(int w=0; w<WPT; w++) {
          if (glb + w < od && k < ld)
            acc[w] += mat_buffer[k][w];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

    }
  }
//    for (int w=0; w<WPT; w++) {
//      if (glb + w < od)
//        output[glb + w] = acc[w];
//    }
}






  //if (act == 0) {
  //  if (output[i] < 0)
  //    output[i] = 0;
  //  else
  //    output[i] = 1;
  //}
  //else if (act == 2) {
  //  output[i] = 1 / (1 + exp(-output[i]));
  //}
  //else if (act == 3) {
  //  output[i] = tanh(output[i]);
  //}
  //else if (act == 4) {
  //  if (output[i] < 0)
  //    output[i] = 0;
  //}
  //else if (act == 5) {
  //  if (output[i] < 0)
  //    output[i] = 0.01*output[i];
 // }
 // else if (act == 6) {
  //   output[i] = exp(output[i]);
 // }
}

##################################################################

#define TS 256
#define WPT 4
#define RTS 8
#define cond (w < WPT) && (glb+w < od)

__kernel void forward(
  const unsigned int ld,
  const unsigned int od,
 __global const float *input,
 __global const float *weights,
 __global const float *bias,
 __global float *output){

  //const int glb = get_global_id(0);
  const int col = get_local_id(0); // Max is TS  32
//  const int row = get_local_id(1); // Max is RTS  8
  const int grp = get_group_id(0);
  const int glb = (grp*TS + col)*WPT; // Max is len(output) / WPT

  const int extra = od % WPT;

  float acc[WPT];
  for (int w=0; cond; w++) {
    acc[w] = bias[glb+w];
  }

  __local float vec_buffer[TS];
  __local float mat_buffer[TS][WPT];

  const int iterations = ld/TS;

  for (int iter=0; iter<=iterations; iter++) { // Runs for iterations+1
    const int tiledIter = TS*iter + col;
    if (tiledIter < ld) { // making sure we don't run off stay withing the size of input
      // Loading input into buffer
      vec_buffer[col] = input[tiledIter];

      // Loading values of the
      for(int w=0; cond; w++) {
        mat_buffer[col][w] = weights[(glb + w)*ld + tiledIter];
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      for (int k=0; k<TS && k<ld; k++) {
        for(int w=0; w<WPT; w++) {
          acc[w] += input[k]*mat_buffer[k][w];
        }
      }
      barrier(CLK_LOCAL_MEM_FENCE);

      for (int w=0; cond; w++) {
        output[glb + w] = acc[w];
      }
    } // assume that iterations*TS-1

//  const int tiledIter = TS*iterations + col;
//  if (tiledIter < ld) {
//
//    vec_buffer[col] = input[tiledIter];
//    for(int w=0; w<WPT; w++) {
//      if (glb+w < ld)
//        mat_buffer[col][w] = weights[(glb + w) * ld + tiledIter];
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int k=0; k < (ld-TS*iterations); k++) {
//      for (int w=0; w<WPT; w++) {
//        acc[w] += input[k] * mat_buffer[k][w];
//      }
//    }
//    barrier(CLK_LOCAL_MEM_FENCE);
//
//    for (int w=0; cond; w++) {
//      output[glb + w] = acc[w];
//    }
  }


}

'''

# // Local memory
#   __local float mat_buffer[BLOCK_SIZE][BLOCK_SIZE];
#   __local float vec_buffer[BLOCK_SIZE];
#   __local float out[BLOCK_SIZE];
#
#   // Thread Identifier
#   int curRow = get_local_id(OUTPUT);
#   int curCol = get_local_id(OUTPUT);
#   int grpRow = get_group_id(OUTPUT);
#   int grpCol = get_group_id(INPUT);
#   int glbRow = get_global_id(OUTPUT);
#   int glbCol = get_global_id(INPUT);
#
#   float acc = 0.0f;
#
#   // copying in the input
#   vec_buffer[curCol] = input[glbCol];
#   mat_buffer[curRow][curCol] = weights[size*(glbRow) + glbCol];
#
#   barrier(CLK_LOCAL_MEM_FENCE);
#
#   for (int row = 0; row < BLOCK_SIZE; row++) {
#     for (int col = 0; col < BLOCK_SIZE; col++) {
#       output[glbRow] += vec_buffer[col] * mat_buffer[row][col];
#     }
#   }