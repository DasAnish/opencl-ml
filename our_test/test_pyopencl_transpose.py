from clobject import *
import pyopencl as cl
import pyopencl.array as pycl_array
import numpy as np
import os
import random

os.chdir('..')
opencl: ClSingleton = ClSingleton.get_instance()
code: Code = Code.get_instance()

## testing tranpose
F = 256
T = 256

# matrix_np = np.array(
    # [[(i*F+j) for j in range(F)] for i in range(T)]
# ).astype(np.float32)
matrix_np = np.random.random(size=(T, F)).astype(np.float32)
matrix_cl = pycl_array.to_device(
    opencl.queue,
    matrix_np
)

input_vec = pycl_array.to_device(
    opencl.queue,
    np.ones(F).astype(np.float32)
)

output_vec = pycl_array.to_device(
    opencl.queue,
    np.zeros(T).astype(np.float32)
)

# print(matrix_cl.shape)
code.program.matrix_vector_mul(
    opencl.queue,
    (T, 8),
    (32, 8),
    np.int32(F),
    np.int32(T),
    input_vec.data,
    matrix_cl.data,
    output_vec.data
)

print("******************python********************")
print(output_vec)
# print("matrix_cl.shape", matrix_cl.shape)
# print(" INPUT VEC", input_vec)
# print("********************************")
# print(" matrix cl", matrix_cl)
# print("********************************")
# print("Outout VEC",output_vec)
print("*********************************")
# matrix_np = np.reshape(matrix_np, (F, T))
output_np = np.matmul(matrix_np, input_vec.get())
print("OTUPUT NP",output_np)
print(output_vec.get() == output_np)


'''

[1. 1. 1. 1. 1. 1. 1. 1.]
********************************
[[ 0.  0.  0.  0.  0.  0.  0.  0.]
 [ 0.  1.  2.  3.  4.  5.  6.  7.]
 [ 0.  2.  4.  6.  8. 10. 12. 14.]
 [ 0.  3.  6.  9. 12. 15. 18. 21.]
 [ 0.  4.  8. 12. 16. 20. 24. 28.]
 [ 0.  5. 10. 15. 20. 25. 30. 35.]
 [ 0.  6. 12. 18. 24. 30. 36. 42.]
 [ 0.  7. 14. 21. 28. 35. 42. 49.]
 [ 0.  8. 16. 24. 32. 40. 48. 56.]
 [ 0.  9. 18. 27. 36. 45. 54. 63.]]
********************************
[  0.  28.  56.  84. 112. 140. 168. 196. 224. 252.]
*********************************
[  0.  28.  56.  84. 112. 140. 168. 196. 224. 252.]

'''
