#define TS 256

__kernel void matrix_vector_mul(const unsigned int size,
 __global const float *input,
 __global const float *weights,
 __global float *output)
{

  const int glb = get_global_id(0);
  const int loc = get_local_id(0);
  const int grp = get_group_id(0);
  float acc = 0.0f;

  __local float vec_buffer[TS];
  __local float mat_buffer[TS];

  const int iterations = size/TS;
  int iter = 0;

  if (iterations) {
    for (int iter = 0; iter < iterations; iter++) {

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
    output[glb] = 0.0f;

    for (int k=0; k<size; k++) {
      output[glb] += input[k]*weights[k];
    }
  }
}

__kernel void weights_del(
    const unsigned int size,
    __global const float *del_values,
    __global const float *z_values,
    __global float *weights_del_values
) {

  const int i = get_global_id(0);
  const int j = get_global_id(1);

  weights_del_values[j*size + i] = del_values[j]*z_values[i];

}

#define BLOCK_DIM 16

__kernel void transpose(
    __global const float *input,
    __global float *output,
    const int width,
    const int height
) {

  __local float block[BLOCK_DIM][BLOCK_DIM];

  const float _height = (float) height;
  const float _width = (float) width;

  const int globalX = get_global_id(0);
  const int globalY = get_global_id(1);
  const int localX = get_local_id(0);
  const int localY = get_local_id(1);

  if (globalX < _width && globalY < _height)
    block[localY][localX] = input[globalY * width + globalX];

  barrier(CLK_LOCAL_MEM_FENCE);

  if (globalX < _width && globalY < _height)
    output[globalX * height + globalY] = block[localY][localX];

}

__kernel void activate(
    __global const float *input,
    __global float *output,
    const unsigned int activation_type
) {

  const int i = get_global_id(0);
  output[i] = input[i];

  if (activation_type == 0) { // BINARY_STEP
    if (output[i] < 0)
      output[i] = 0;
    else
      output[i] = 1;
  }
  else if (activation_type == 2) { // SIGMOID
    output[i] = 1 / (1 + exp(-output[i]));
  }
  else if (activation_type == 3) { // TANH
    output[i] = tanh(output[i]);
  }
  else if (activation_type == 4) { // RELU
    if (output[i] < 0)
      output[i] = 0;
  }
  else if (activation_type == 5) { // LEAKY_RELU
    if (output[i] < 0)
      output[i] = 0.01*output[i];
  }
  else if (activation_type == 6) { // SOFTMAX
     output[i] = exp(output[i]);
  }

}