import torch
import numpy as np


def validate_init_args(at_target, at_target_type, num_labels,
                        loss_type, dual_factor, is_auc, num_anchors):

    if not isinstance(num_labels, int):
        raise TypeError("num_labels must be an integer.")
    if num_labels < 1:
        raise ValueError("num_labels must be at least one.")

    if not isinstance(dual_factor, float):
        raise TypeError("dual factor must be a float.")

    if loss_type not in ['cross_entropy', 'hinge']:
        raise TypeError(
            "loss_type must be either 'xent' or 'hinge'."
        )

    if is_auc:
        if not isinstance(num_anchors, int):
            raise TypeError("num_anchors must be an integer.")
        if num_anchors < 1:
            raise ValueError("num_anchors must be at least one.")
        if not isinstance(at_target, tuple):
            raise TypeError("at_target must be a tuple.")
        if len(at_target) != 2:
            raise TypeError("at_target must be a tuple of size 2")
        if not isinstance(at_target[0], float):
            raise TypeError("at_target[0] must be a float.")
        if not isinstance(at_target[1], float):
            raise TypeError("at_target[1] must be a float.")
        if not 0.0 <= at_target[0] <= at_target[1] <= 1.0:
            raise ValueError(
                "{} values must follow 0 <= {} <= {} <= 1.".format(
                    at_target_type,
                    at_target[0], at_target[1]
                )
            )
    else:
        if not isinstance(at_target, float):
            raise TypeError("{} must be a float.".format(
                at_target_type)
            )
        if not 0 <= at_target <= 1:
            raise ValueError(
                "at_target mustn't be smaller than zero or greater "
                "than 1."
            )


def validate_prepare_logits_labels_weights(logits, labels,
                                           weights, num_labels):

    if logits.dim() == 1:
        logits = logits.unsqueeze(-1)

    if labels.dim() == 1:
        labels = labels.unsqueeze(-1)

    N = logits.size()[0]

    if labels.size()[0] != N:
        raise ValueError(
            "size mismatch: labels size is: {} while "
            "logits size is: {}.".format(labels.size(), logits.size())
        )

    if logits.size()[1] != num_labels:
        raise ValueError(
            "num of labels is {} while logits is either a 1-d tensor or"
            "a tensor of shape: {}".format(num_labels, logits.size())
        )

    if labels.size()[1] != num_labels:
        raise ValueError(
            "num of labels is {} while labels is either a 1-d tensor or"
            "a tensor of shape: {}".format(num_labels, labels.size())
        )

    if weights is None:
        weights = torch.ones_like(logits)

    if weights.dim() > 2:
        raise ValueError("weights must be either a 1-d or "
                         "a 2-d tensor")

    if weights.dim() == 2 and weights.size() != (N, num_labels):
        raise ValueError(
            "weights given as a 2-d tensor do not match the shape "
            "of logits/labels (expected: {}, found: {})".format(
                logits.size(), weights.size()
            )
        )
    if weights.dim() == 1:
        if weights.size()[0] != N:
            raise ValueError(
                "weights given as a 1-d tensor do not match the shape "
                "of logits/labels height (expected: ({},) found: {})".format(
                    N, weights.size()
                )
            )
        else:
            weights = weights.unsqueeze(-1)

    return logits, labels, weights


def prepare_positive_negative_weights(positive_weights, negative_weights,
                                      required_shape, device):

    if np.isscalar(positive_weights):
        positive_weights = torch.empty(required_shape).fill_(
            positive_weights
        )
        positive_weights = positive_weights.to(device)

    if np.isscalar(negative_weights):
        negative_weights = torch.empty(required_shape).fill_(
            negative_weights
        )
        negative_weights = negative_weights.to(device)

    return positive_weights, negative_weights


def build_anchors(target_range, num_anchors):

    target_values = np.linspace(start=target_range[0],
                                stop=target_range[1],
                                num=num_anchors + 2)[1:-1]
    delta = target_values[0] - target_range[0]

    return torch.FloatTensor(target_values), delta


def build_label_priors(labels, label_priors=None, weights=None,
                       positive_pseudocount=1.0, negative_pseudocount=1.0):

    if label_priors is not None:
        if label_priors.size() != (labels.size()[1],):
            raise ValueError("label priors must")
        else:
            return label_priors

    weighted_label_counts = (weights * labels).sum(0)
    weight_sum = weights.sum(0)
    label_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )
    return label_priors


def weighted_hinge_loss(labels, logits,
                        positive_weights=1.0,
                        negative_weights=1.0):

    positive_term = (1.0 - logits).clamp(min=0) * labels
    negative_term = (1.0 + logits).clamp(min=0) * (1.0 - labels)

    return (
            positive_term * positive_weights
            + negative_term * negative_weights
    )


def weighted_cross_entropy_loss(labels,
                           logits,
                           positive_weights=1.0,
                           negative_weights=1.0):

    softerm_plus = (-logits).clamp(min=0) + \
                   torch.log(1.0 + torch.exp(-torch.abs(logits)))

    weight_dependent_factor = (
        negative_weights + (positive_weights - negative_weights)*labels
    )

    return (negative_weights * (logits - labels * logits) +
            weight_dependent_factor * softerm_plus)
