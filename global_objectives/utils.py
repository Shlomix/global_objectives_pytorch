import torch
import numpy as np


def FloatTensor(*args):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)



class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        #return input.clamp(min=0)
        return torch.abs(input)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def lagrange_multiplier(x):
    return LagrangeMultiplier.apply(x)


def one_hot_encoding(logits, targets):
    N, C = logits.size()
    _targets = targets.unsqueeze(-1)
    if C > 1:
        labels = FloatTensor(N, C).zero_().scatter(1, _targets.long().data, 1)
    else:
        labels = _targets
    return labels


