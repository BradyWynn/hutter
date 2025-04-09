import numpy as np

with open('enwik9', 'r', encoding='utf-8') as f:
    text = f.read()

# print(set(text))
# print(f"number of unique characters: {len(set(text))}")
# print(f"length: {len(text)}")

a = np.array(list(text.encode("utf-8")))
np.save("enwik9.npy", a)