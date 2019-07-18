import torch


def FloatTensor(*args):
    if torch.cuda.is_available():
        return torch.cuda.FloatTensor(*args)
    else:
        return torch.FloatTensor(*args)


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dual_factor=1.0):
        ctx.save_for_backward(input)
        ctx.dual_factor = dual_factor
        return torch.abs(input)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.dual_factor*grad_output.neg(), None


def lagrange_multiplier(_lambda, dual_factor=1.0):
    return LagrangeMultiplier.apply(_lambda, dual_factor)


def one_hot_encoding(logits, targets):
    N, C = logits.size()
    _targets = targets.unsqueeze(-1)
    if C > 1:
        labels = FloatTensor(N, C).zero_().scatter(1, _targets.long().data, 1)
    else:
        labels = _targets
    return labels


