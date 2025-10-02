import numpy as np
from bpeasy.tokenizer import BPEasyTokenizer

with open('enwik9', 'r') as f:
    text = f.read()

tokenizer = BPEasyTokenizer.from_file("tokenizer.json")
np.save('tokens.npy', np.array(tokenizer.encode(text)))

# enc = tiktoken.get_encoding("gpt2")
# tokens = enc.encode(text)
# a = np.array(tokens)
# np.save("enwik9.npy", a)

# print(set(text))
# print(f"number of unique characters: {len(set(text))}")
# print(f"length: {len(text)}")

# a = np.array(list(text.encode("utf-8")))
# np.save("enwik9.npy", a)