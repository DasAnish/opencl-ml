import numpy as np
import time

N = 10000
M = 10000
layer1 = np.random.uniform(-1, 1, N)
weights12 = np.random.uniform(-1, 1, size=(N, M))
weights23 = np.random.uniform(-1, 1, size=(M, N))
weights34 = np.random.uniform(-1, 1, size=(N,M))


def forward():
    layer = np.random.uniform(-1, 1, N)
    layer = np.matmul(layer, weights12)
    layer = np.matmul(layer, weights23)
    layer = np.matmul(layer, weights34)

s =0
A=10
for _ in range(A):
    start = time.time()
    forward()
    stop = time.time()
    s += stop - start

print(f'Average = {s} / {A} {s/A}s   {N},{M}')