import lzma
import bz2
import numpy as np
import matplotlib.pyplot as plt

def compress(x, b, e):
    return 2**e * np.round(np.minimum(np.maximum(2**-(e)*x, -2**(b-1)), 2**(b-1)-1))

x = np.random.randn(512, 512).astype(np.float32)

for i in np.linspace(-8, 5, 100):
    y = compress(x, 8, i)

    b = lzma.compress(y)
    print(len(b), i)
# a = lzma.compress(x)
# print(len(a))
# b = lzma.compress(y)
# print(len(b))

# a = lzma.compress(np.load("w.npy"))
# print(len(a))
# b = bz2.compress(np.load("w.npy"))
# print(len(b))