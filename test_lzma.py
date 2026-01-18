import os
import lzma
import bz2
import numpy as np
import matplotlib.pyplot as plt
import torch

sum_list = []

for file in os.listdir("model"):
    if "weight" in file:
        w = torch.load(os.path.join("model", file))
        z = torch.load(os.path.join("model", f"{file[:-9]}z.pt"))

        # print(w.shape, z.shape)

        wz = torch.masked_select(w.flatten(), z.round().flatten().bool())

        b = lzma.compress(np.packbits(z.numpy().astype(np.bool)))
        a = lzma.compress(wz.numpy().astype(np.float16))
        c = lzma.compress((w*z.round()).numpy().astype(np.float16))
        # print(len(a) + len(b), len(c), file)
        sum_list.append(len(a) + len(b))
print(sum(sum_list))

# wz = w * z.round()

# plt.imshow(wz.round())
# plt.show()

# print(w.shape)
# print(z.shape)

# weights = []

# for file in os.listdir("model"):
#     w = torch.load(os.path.join("model", file)).flatten()
#     print(file)
#     plt.hist(w.flatten(), 100, density=True, log=True)
#     plt.show()
    # weights.append(torch.load(os.path.join("model", file)).flatten())

# weights = torch.cat(weights)
# print(weights.shape)
# a = lzma.compress(weights.numpy())
# print(len(a))

# def compress(x, b, e):
#     return 2**e * np.round(np.minimum(np.maximum(2**-(e)*x, -2**(b-1)), 2**(b-1)-1))

# x = np.random.randn(512, 512).astype(np.float32)

# for i in np.linspace(-8, 5, 100):
#     y = compress(x, 8, i)

#     b = lzma.compress(y)
#     print(len(b), i)
# a = lzma.compress(x)
# print(len(a))
# b = lzma.compress(y)
# print(len(b))

# a = lzma.compress(np.load("w.npy"))
# print(len(a))
# b = bz2.compress(np.load("w.npy"))
# print(len(b))