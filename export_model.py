import os
import torch
import numpy as np

file = torch.load('model_11899.pt', map_location=torch.device('cpu'), weights_only=False)
model = file['model']
for i in model.keys():
    np.save(os.path.join('model', i), model[i].float().numpy())
    print(i)