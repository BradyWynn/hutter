import numpy as np
import tiktoken

with open('enwik9', 'r', encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)
a = np.array(tokens)
np.save("tokenized_enwik9.npy", a)