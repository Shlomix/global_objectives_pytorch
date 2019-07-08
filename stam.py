import torch
import numpy as np


def FloatTensor(*args):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)
    
    
a = FloatTensor(5,3)

print(a)

b = FloatTensor(a.shape)

print(b)
