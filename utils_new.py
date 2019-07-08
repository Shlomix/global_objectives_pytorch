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
        return input.clamp(min=0)

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


def calc_class_priors(labels,
                      class_priors=None, weights=None,
                      positive_pseudocount=1.0,
                      negative_pseudocount=1.0):

    if class_priors is not None:
        return class_priors

    weighted_label_counts = (weights * labels).sum(0)
    weight_sum = weights.sum(0)
    class_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )

    return class_priors


def calc_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):

    if not torch.is_tensor(positive_weights):
        if not np.isscalar(positive_weights):
            raise ValueError(
                "positive_weights must be either a scalar or a tensor"
            )
        elif torch.is_tensor(negative_weights):
            positive_weights = FloatTensor(negative_weights.shape).fill_(positive_weights)

    elif not torch.is_tensor(negative_weights):
        if not np.isscalar(negative_weights):
            raise ValueError(
                "negative_weights must be either a scalar or a tensor"
            )
        else:
            negative_weights = FloatTensor(positive_weights.shape).fill_(negative_weights)

    elif positive_weights.size() != negative_weights.size():
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            .format(positive_weights.size(), negative_weights.size())
        )

    if positive_weights.dim() != 2:
        raise ValueError(
            "number of dimensions for positive weights "
            "must be 2 "
            "but found {} instead."
            .format(positive_weights.dim())
        )

    positive_term = (1.0 - logits).clamp(min=0) * labels
    negative_term = (1.0 + logits).clamp(min=0) * (1.0 - labels)

    return (
        positive_term.unsqueeze(-1) * positive_weights
        + negative_term.unsqueeze(-1) * negative_weights
    )


def build_anchors(target_range, num_anchors):

    if len(target_range) != 2:
        raise ValueError("length of precision_range {:d} must be 2"
                         .format(target_range)
                         )

    if not 0 <= target_range[0] <= target_range[1] <= 1:
        raise ValueError("target values must follow "
                         "0 <= {:f} <= {:f} <= 1"
                         .format(target_range[0], target_range[1])
                         )

    target_values = np.linspace(start=target_range[0],
                                stop=target_range[1],
                                num=num_anchors + 1)[1:]

    delta = (target_values[1] - target_values[0]) / num_anchors

    return FloatTensor(target_values), delta
