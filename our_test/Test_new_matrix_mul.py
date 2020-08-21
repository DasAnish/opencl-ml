import pyopencl as cl  # Import the OpenCL GPU computing API
import pyopencl.array as pycl_array  # Import PyOpenCL Array (a Numpy array plus an OpenCL buffer object)
import numpy as np  # Import Numpy number tools
import time
import threading

context = cl.create_some_context()  # Initialize the Context
queue = cl.CommandQueue(context)  # Instantiate a Queue

N=256
M=256
_input = np.arange(N)
input_layer = pycl_array.to_device(queue, np.where(_input%2, 1, 0).astype(np.float32))
print(np.sum(input_layer.get()))
output_layer = pycl_array.to_device(queue, np.zeros(M).astype(np.float32))
matrix = pycl_array.to_device(queue, np.ones(M*N).astype(np.float32))
bias = pycl_array.to_device(queue, np.zeros_like(output_layer).astype(np.float32))

mat = np.arange(M)
for i in range(M-1):
    mat = np.concatenate([mat, np.arange(M)])
mat = mat.astype(np.float32)
matrix.set(mat)


# because = pycl_array.to_device(queue, np.ones(256).astype(np.int32))

old_code= '''
#define TS 256

__kernel void forward(const unsigned int size,
 __global const float *input,
 __global const float *weights,
 __global const float *bias,
 __global float *output)
{

  const int glb = get_global_id(0);
  const int loc = get_local_id(0);
  const int grp = get_group_id(0);
  float acc = bias[glb];

  __local float vec_buffer[TS];
  __local float mat_buffer[TS];

  const int iterations = size/TS;
  int iter = 0;

  if (iterations) {
    for (; iter < iterations; iter++) {

      const int tiledIter = TS*iter + loc;
      vec_buffer[loc] = input[tiledIter];
      mat_buffer[loc] = weights[glb*size + tiledIter];

      barrier(CLK_LOCAL_MEM_FENCE);

      for (int k=0; k<TS; k++) {
        acc += vec_buffer[k]*mat_buffer[k];
      }

    }
    barrier(CLK_LOCAL_MEM_FENCE);
    output[glb] = acc;

  } else {
    output[glb] = bias[glb];

    for (int k=0; k<size; k++) {
      output[glb] += input[k]*weights[k];
    }
  }
}'''

with open('..\\kernel.cl', 'r') as f:
    code = f.read()
# print(code)
# program = cl.Program(context, code).build()
# old_program = cl.Program(context, old_code).build()
#
# print(output_layer)
# print(input_layer)
# print(matrix)
# program.forward(queue, output_layer.shape, None, np.int32(N),
#                 input_layer.data, matrix.data, bias.data, output_layer.data)

# print(output_layer)

matrix_np = matrix.get().reshape(N, M)
# # print(matrix_np.shape)
# bias_np = bias.get()


def old(layer, program, start):
    input_layer.set(layer)
    program.forward(queue, output_layer.shape, None, np.int32(N),
                    input_layer.data, matrix.data, bias.data, output_layer.data)
    end = time.time()
    return end-start


def opencl(layer, program, start):

    input_layer.set(layer)
    program.forward(queue, (M,), (256,), np.int32(N),
                    input_layer.data, matrix.data, bias.data, output_layer.data)
    # output =
    end=time.time()
    return end-start


def NP(layer):
    output_layer.set(np.matmul(matrix_np, layer))


so = 0
sn = 0

for _ in range(100):
    l=np.ones(N).astype(np.float32)
    # print(l)
    # sn += old(l, cl.Program(context, old_code).build(), time.time())
    # print(f'Old_code: {_} {sn} o={output_layer[0]}')
    # so += opencl(l, cl.Program(context, code).build(), time.time())
    # print(f'Opencl:   {_} {so} o=\n{output_layer}')

    NP(l)
    print(output_layer)

    print('Actual Value:          \t\t\t  ', np.sum(l))
    print("###############################################################################")
    break
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