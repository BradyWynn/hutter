import numpy as np
# import tiktoken

with open('enwik9', 'rb') as f:
    text = f.read()

# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode(text)
# a = np.array(tokens)
# np.save("tokenized_enwik9.npy", a)

# print(set(text))
# print(f"number of unique characters: {len(set(text))}")
# print(f"length: {len(text)}")

# a = np.array(list(text.encode("utf-8")))
# np.save("enwik9.npy", a)

a = np.frombuffer(text, dtype=np.uint8)
np.save('enwik9.npy', a)
