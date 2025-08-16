import torch
import numpy as np
from model import GPT
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm
import tiktoken

model = GPT()
model.load_state_dict(torch.load("model_11899.pt", map_location=torch.device('cpu'), weights_only=False)['model'])
# model = model.bfloat16()

enc = tiktoken.get_encoding("gpt2")

str_context = "The industrial revolution and its consequences have been a disaster for the human race."
tokens = torch.tensor(enc.encode(str_context), dtype=torch.long)

print(str_context, end="")

with torch.no_grad():
    context = tokens[:512]

    model.setup_caches(max_batch_size=1, max_seq_length=1024)

    # prefill
    logits = model(context.unsqueeze(0), torch.arange(len(context)))
    probs = F.softmax(logits, dim=-1)[0, -1]

    input_pos = torch.tensor([len(context)])
    next_token = torch.multinomial(probs.squeeze(), 1)

    for i in range(512):
        print(enc.decode([next_token]), end="")
        logits = model(next_token.unsqueeze(0), input_pos)
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs.squeeze(), 1)
        str_context += chr(next_token)
        input_pos += 1