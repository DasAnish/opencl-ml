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
}