#define TS 8
#define WPT 32
#define ROW_DIM 0
#define COL_DIM 1

__kernel void matrix_vector_mul(
    const unsigned int n,
    const unsigned int reset_output,
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
  if (reset_output) output[glb_row] = 0.0f;

  for (int it = 0; it < n/TS+1; it++) {

    const int tiledIter = TS*it + loc_col;
    if (tiledIter < n) {
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

__kernel void weights_del(
    const unsigned int size,
    __global const float *del_values,
    __global const float *z_values,
    __global float *weights_del_values
) {

  const int i = get_global_id(0); // max is layer_size
  const int j = get_global_id(1); // max is next_layer_size

  weights_del_values[j*size + i] += del_values[j]*z_values[i];//j*i;//

}

// ACTIVATION FUNCTIONS

float sigmoid(float v) {
  return 1.0 / (1.0 + exp(-v));
}

__kernel void activate(
    __global float *input,
    __global float *output,
    const unsigned int activation_type
) {

  const int i = get_global_id(0);

  if (activation_type == 0) { // BINARY_STEP
    if (input[i] < 0)
      input[i] = 0;
    else
      input[i] = 1;
  }
  else if (activation_type == 2) { // SIGMOID
    output[i] = sigmoid(input[i]);
  }
  else if (activation_type == 3) { // TANH
    output[i] = tanh(input[i]);
  }
  else if (activation_type == 4) { // RELU
    if (input[i] < 0)
      output[i] = 0;
    else
      output[i] = input[i];
  }
  else if (activation_type == 5) { // LEAKY_RELU
    if (input[i] < 0)
      output[i] = 0.01*output[i];
    else
      output[i] = input[i];
  }
  else if (activation_type == 6) { // SOFTMAX
//     output[i] = exp(output[i]);
  }

}

__kernel void activation_derivative (
    __global const float* activation,
    __global float* output,
    const unsigned int activation_type
) {

  const int i = get_global_id(0);

  if (activation_type == 1) { // LINEAR
//    output[i] *= 1;
  }
  else if (activation_type == 2) { // SIGMOID
    output[i] *= activation[i] * (1 + activation[i]);
  }
  else if (activation_type == 3) { // TANH
    float v = cosh(activation[i]);
    output[i] *= (1.0 / (v * v));
  }
  else if (activation_type == 4) { // RELU
    if (activation[i] < 0)
      output[i] *= 0;
    else
      output[i] *= 1;
  }
  else if (activation_type == 5) { // LEAKY_RELU
    if (activation[i] < 0)
      output[i] *= 0.01;
    else
      output[i] *= 1;
  }
  else if (activation_type == 6) { // SOFTMAX
     output[i] *= activation[i] * (1 - activation[i]);
  }

}