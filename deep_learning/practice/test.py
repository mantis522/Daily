import torch
import numpy as np

x = np.array([[0], [1], [2]])

print(x.shape)
print(x)

y = torch.Tensor(x)
print(y.shape)
print(y)

y = torch.unsqueeze(y, 1)
print(y.shape)
print(y)