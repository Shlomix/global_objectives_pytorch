import torch
import numpy as np


def build_class_priors(labels,
                      class_priors=None,
                      weights=None,
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


def calc_hinge_loss(labels,
                    logits,
                    positive_weights=1.0,
                    negative_weights=1.0):

    if not torch.is_tensor(positive_weights) and not np.isscalar(positive_weights):
        raise ValueError(
            "positive_weights must be either a scalar or a tensor"
        )
    if not torch.is_tensor(negative_weights) and not np.isscalar(negative_weights):
        raise ValueError(
            "negative_weights must be either a scalar or a tensor"
        )
    if torch.is_tensor(positive_weights) and np.isscalar(negative_weights):
        negative_weights = torch.zeros_like(positive_weights).data.fill_(negative_weights)

    elif torch.is_tensor(negative_weights) and np.isscalar(positive_weights):
        positive_weights = torch.zeros_like(negative_weights).data.fill_(positive_weights)

    elif positive_weights.size() != negative_weights.size():
        raise ValueError(
            "shape of positive_weights and negative_weights "
            "must be the same! "
            "shape of positive_weights is {0}, "
            "but shape of negative_weights is {1}"
            .format(positive_weights.size(), negative_weights.size())
        )

    positive_term = (1.0 - logits).clamp(min=0) * labels
    negative_term = (1.0 + logits).clamp(min=0) * (1.0 - labels)

    if positive_weights.dim() not in [1, 2]:
        raise ValueError(
            "number of dimensions for "
            "positive and negative weights"
            "must be 2 but found {} instead."
                .format(positive_weights.dim())
        )

    return (
            positive_term * positive_weights
            + negative_term * negative_weights
    )


def build_anchors(target_range, num_anchors, target_type='target'):
    if len(target_range) != 2:
        raise ValueError("length of {} {:d} must be 2"
                         .format(target_type, target_range)
                         )

    if not 0 <= target_range[0] <= target_range[1] <= 1:
        raise ValueError("{} values must follow "
                         "0 <= {:f} <= {:f} <= 1"
                         .format(target_type,
                                 target_range[0],
                                 target_range[1])
                         )

    target_values = np.linspace(start=target_range[0],
                                stop=target_range[1],
                                num=num_anchors + 2)[1:-1]

    delta = (target_values[0] - target_range[0])

    return torch.tensor(target_values), delta


def one_hot_encoding(logits, targets):
    N, C = logits.size()
    _targets = targets.unsqueeze(-1)
    if C > 1:
        labels = torch.zeros_like(logits).scatter(1, _targets.long().data, 1)
    else:
        labels = _targets
    return labels


