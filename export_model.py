import os
import torch
import numpy as np

file = torch.load('log/model_02579.pt', map_location=torch.device('cpu'), weights_only=False)
model = file['model']
for i in model.keys():
    torch.save(model[i], f"{os.path.join('model', i)}.pt")
    print(i)
    # np.save(os.path.join('model', i), model[i].float().numpy())
    # print(i)