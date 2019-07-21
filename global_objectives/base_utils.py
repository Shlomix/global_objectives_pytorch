import torch
import numpy as np


def validate_init_args(at_target,
                       at_target_type,
                       num_labels,
                       dual_factor,
                       is_auc,
                       num_anchors):

    if not isinstance(num_labels, int):
        raise TypeError("num_labels must be an integer.")
    if num_labels < 1:
        raise ValueError("num_labels must be at least one.")

    if not isinstance(dual_factor, float):
        raise TypeError("dual factor must be a float.")

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
                "{} values must follow"
                "0 <= {} <= {} <= 1.".format(
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
                "at_target mustn't be smaller than zero "
                "or greater than 1."
            )


def validate_prepare_logits_labels_weights(logits,
                                            labels,
                                            weights,
                                            num_labels):

    #validate & prepare logits
    if logits.dim() == 1:
        if num_labels == 1:
            logits = logits.unsqueeze(-1)
        else:
            raise ValueError(
                "num of labels is {} while "
                "logits is 1-d vector.".format(
                    num_labels
                )
        )
    elif logits.dim() == 2 and logits.size()[1] != num_labels:
        raise ValueError(
            "num of labels is {} while "
            "logits width is {}."
            .format(num_labels, logits.size()[1])
        )

    N = logits.size()[0]

    #validate & prepare labels
    if labels.size() == (N,):
        _labels = labels.unsqueeze(-1)
        if num_labels > 1:
            labels = torch.zeros_like(logits).scatter(
                1, _labels.long().data, 1
            )
        else:
            labels = _labels
    else:
        raise ValueError(
            "targets must be of size: ({},) "
            "found {} instead"
            .format(N, labels.size())
        )

    #validate & prepare weights
    if weights is None:
        weights = torch.FloatTensor(N).data.fill_(1.0)
    else:
        if not torch.is_tensor(weights):
            raise ValueError(
                "weights must be a tensor"
            )
        if weights.dim() > 2:
            raise ValueError(
                "weights must be either a "
                "1-d or a 2-d tensor."
            )
        elif weights.dim() == 2 and weights.size() != logits.size():
            raise ValueError(
                "weights given in 2-d must be of shape "
                "(logits_height, num_labels) = ({},{}) "
                "but found: {} instead.".format(
                    N, num_labels, weights.size()
                )
            )
        elif weights.size()[0] != N:
            raise ValueError(
                "weights given in 1-d must be of shape "
                "(logits_height,) but found: "
                "({},) instead.".format(
                    logits.size(), weights.size()
                )
            )

    if weights.dim() == 1:
        weights.unsqueeze_(-1)
    weights = weights.to(logits.device)

    return logits, labels, weights


def _validate_prepare_positive_negative_weights(positive_weights,
                                                negative_weights,
                                                required_shape,
                                                device):

    if np.isscalar(positive_weights):
        positive_weights = torch.FloatTensor(required_shape).fill_(
            positive_weights).to(device)

    elif torch.is_tensor(positive_weights):
        if positive_weights.size() != required_shape:
            raise ValueError(
                "positive_weights must be of size {} "
                "but found size: {} instead.".format(
                required_shape, positive_weights.size()
                )
            )
    else:
        raise ValueError(
            "positive_weights must be either a "
            "scalar or a tensor"
        )

    if np.isscalar(negative_weights):
        negative_weights = torch.FloatTensor(required_shape).fill_(
            negative_weights).to(device)

    elif torch.is_tensor(negative_weights):
        if negative_weights.size() != required_shape:
            raise ValueError(
                "positive_weights must be of size {} "
                "but found size: {} instead.".format(
                required_shape, negative_weights.size()
                )
            )
    else:
        raise ValueError(
            "positive_weights must be either a "
            "scalar or a tensor"
        )

    return positive_weights, negative_weights


def _validate_lambda_term(lambda_term, lambdas):
    if lambda_term.size() != lambdas.size():
        raise ValueError(
            "lambda term must be of size {} but "
            "found {} instead".format(
                lambdas.size(),
                lambda_term.size()
            )
        )


def build_anchors(target_range, num_anchors):

    target_values = np.linspace(start=target_range[0],
                                stop=target_range[1],
                                num=num_anchors + 2)[1:-1]
    delta = (target_values[0] - target_range[0])

    return torch.FloatTensor(target_values), delta


def build_label_counts_priors(labels,
                              label_priors=None,
                              weights=None,
                              positive_pseudocount=1.0,
                              negative_pseudocount=1.0):

    if label_priors is not None:
        return label_priors

    weighted_label_counts = (weights * labels).sum(0)
    weight_sum = weights.sum(0)
    label_priors = torch.div(
        weighted_label_counts + positive_pseudocount,
        weight_sum + positive_pseudocount + negative_pseudocount,
    )
    return weighted_label_counts, label_priors


def weighted_hinge_loss(labels,
                        logits,
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


