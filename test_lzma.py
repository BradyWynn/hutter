import lzma
import bz2
import numpy as np


a = lzma.compress(np.load("w.npy"))
print(len(a))
b = bz2.compress(np.load("w.npy"))
print(len(b))