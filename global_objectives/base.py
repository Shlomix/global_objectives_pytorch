import torch
import torch.nn as nn
import numpy as np
import global_objectives.utils as utils
from global_objectives.utils import FloatTensor


class BaseLoss(nn.Module):

    def __init__(self, target_type,
                 target=None, num_classes=1,
                 auc=False, num_anchors=1):

        nn.Module.__init__(self)

        self.target_type = target_type
        self.num_classes = num_classes
        self.auc = auc
        self.num_anchors = num_anchors

        if self.auc:

            if target is None:
                self.target_range = (0.0, 1.0)

            self.target_range = target

            self.target_values, self.delta = BaseLoss.build_anchors(
                self.target_range, self.num_anchors
            )

            self.biases = nn.Parameter(
                FloatTensor(self.num_classes, self.num_anchors).zero_()
            )
            self.lambdas = nn.Parameter(
                FloatTensor(self.num_classes, self.num_anchors).data.fill_(
                    1.0
                )
            )

        else:
            if target is None:
                raise ValueError("{} should be directly specified".format(target_type))

            elif target < 0.0 or target > 1.0:
                raise ValueError("{} must be in the range [0.0,1.0]".format(target_type))

            self.target = target

            self.biases = nn.Parameter(
                FloatTensor(self.num_classes).zero_()
            )
            self.lambdas = nn.Parameter(
                FloatTensor(self.num_classes).data.fill_(
                    1.0
                )
            )

    def forward(self, logits, targets,
                reduce=True, size_average=True,
                weights=None):

        if logits.dim() == 1:
            logits = logits.unsqueeze(-1)

        N, C = logits.size()

        if self.num_classes != C:
            raise ValueError(
                "num classes is %d while logits width is %d" % (self.num_classes, C)
            )

        labels = utils.one_hot_encoding(logits, targets)

        if weights is None:
            weights = FloatTensor(N).data.fill_(1.0)

        if weights.dim() == 1:
            weights.unsqueeze_(-1)

        # class_priors' shape = [C]
        class_priors = BaseLoss.calc_class_priors(labels, weights=weights)
        # lambdas' shape = [C, K] or [C]
        lambdas = utils.lagrange_multiplier(self.lambdas)
        # positive(negative)_weights' shape = [C, K]
        positive_weights, negative_weights = self.calc_pos_neg_weights(lambdas)
        # lambda_term's shape = [C, K] (auc) or [C]
        lambda_term = self.calc_lambda_term(lambdas, class_priors)

        if self.auc:
            # hinge_loss's shape = [N,C,K]
            hinge_loss = self.calc_hinge_loss(
                labels.unsqueeze(-1),
                logits.unsqueeze(-1),
                positive_weights=positive_weights,
                negative_weights=negative_weights,
            )

            per_anchor_loss = weights.unsqueeze(-1) * hinge_loss - lambda_term
            # loss = per_label_loss
            loss = per_anchor_loss.sum(2) * self.delta
            loss /= (self.target_range[1] - self.target_range[0] - self.delta)

        else:
            # hinge_loss's shape = [N,C]
            hinge_loss = BaseLoss.calc_hinge_loss(
                labels,
                logits,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
            )
            loss = weights * hinge_loss - lambda_term

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    def calc_lambda_term(self, lambdas, class_priors):
        raise NotImplementedError

    def calc_pos_neg_weights(self, lambdas):
        raise NotImplementedError

    @staticmethod
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

    @staticmethod
    def calc_hinge_loss(labels, logits, positive_weights=1.0, negative_weights=1.0):

        if not torch.is_tensor(positive_weights) and not np.isscalar(positive_weights):
                raise ValueError(
                    "positive_weights must be either a scalar or a tensor"
                )

        if not torch.is_tensor(negative_weights) and not np.isscalar(negative_weights):
                raise ValueError(
                    "negative_weights must be either a scalar or a tensor"
                )

        if torch.is_tensor(positive_weights) and np.isscalar(negative_weights):
            negative_weights = FloatTensor(positive_weights.shape).fill_(negative_weights)

        elif torch.is_tensor(negative_weights) and np.isscalar(positive_weights):
            positive_weights = FloatTensor(negative_weights.shape).fill_(positive_weights)

        # if both are tensors to begin with.
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


    @staticmethod
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
                                    num=num_anchors + 1)[1:]

        delta = (target_values[1] - target_values[0]) / num_anchors

        return FloatTensor(target_values), delta
