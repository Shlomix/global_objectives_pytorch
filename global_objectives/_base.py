import torch
import torch.nn as nn
import global_objectives.base_utils as base_utils
from global_objectives.utils import FloatTensor


class LagrangeMultiplier(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dual_factor):
        ctx.save_for_backward(input)
        ctx.dual_factor = dual_factor
        return torch.abs(input)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.dual_factor*grad_output.neg(), None


def lagrange_multiplier(_lambda, dual_factor=1.0):
    return LagrangeMultiplier.apply(_lambda, dual_factor)


class BaseLoss(nn.Module):

    def __init__(self,
                 at_target,
                 at_target_type,
                 num_labels,
                 dual_factor=0.1,
                 is_auc=False,
                 num_anchors=20):
        nn.Module.__init__(self)

        if not isinstance(num_labels, int):
            raise TypeError("num_labels must be an integer.")
        if num_labels < 1:
            raise ValueError("num_labels must be at least one.")
        self.num_labels = num_labels

        if not isinstance(dual_factor, float):
            raise TypeError("dual factor must be a float.")
        self.dual_factor = dual_factor

        if at_target_type not in ["tpr", "fpr", "precision", "recall"]:
            raise TypeError(
                "at_target_type must be one of the following:"
                "(tpr, fpr, precision, recall)."
            )
        self.at_target_type = at_target_type
        self.is_auc = is_auc

        if self.is_auc:
            if not isinstance(num_anchors, int):
                raise TypeError("num_anchors must be an integer.")
            if num_anchors < 1:
                raise ValueError("num_anchors must be at least one.")
            self.num_anchors = num_anchors

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
                    "at_target values must follow"
                    "0 <= {} <= {} <= 1."
                    .format(at_target[0], at_target[1])
                )
            self.at_target_range = at_target

            _at_target_values, self.delta = base_utils.build_anchors(
                self.at_target_range, self.num_anchors
            )
            self.register_buffer('at_target_values', _at_target_values)
            self.biases = nn.Parameter(
                FloatTensor(self.num_labels, self.num_anchors).zero_()
            )
            self.lambdas = nn.Parameter(
                FloatTensor(self.num_labels, self.num_anchors).data.fill_(
                    1.0
                )
            )
        else:
            if not isinstance(at_target, float):
                raise TypeError("at_target must be a float.")
            if not 0 <= at_target <= 1:
                raise ValueError(
                    "at_target mustn't be smaller than zero "
                    "or greater than 1."
                )
            self.at_target = at_target
            self.biases = nn.Parameter(
                torch.FloatTensor(self.num_labels).zero_()
            )
            self.lambdas = nn.Parameter(
                torch.FloatTensor(self.num_labels).data.fill_(
                    1.0
                )
            )

    def forward(self,
                logits,
                targets,
                class_priors=None,
                weights=None,
                reduce=True,
                size_average=True):

        # logits priors' shape = [N, C]
        # after "prepare_logits_labels_weights() method":
        #     labels' shape = [N, C]
        #     weights' shape = [N, 1] or [N, C]
        logits, labels, weights = \
            base_utils.validate_prepare_logits_labels_weights(
                logits=logits, targets=targets,
                weights=weights, num_labels=self.num_labels
        )
        # labels priors' shape = [C]
        label_priors, _ = base_utils.build_label_counts_priors(
            labels=labels, weights=weights
        )
        # lambdas' shape = [C, K] or [C]
        lambdas = lagrange_multiplier(
            _lambda=self.lambdas, dual_factor=self.dual_factor
        )
        # lambda_term's shape = [C, K] (auc) or [C]
        lambda_term = self.get_lambda_term(
            lambdas=lambdas,
            targets=self.at_target_values if self.is_auc else self.at_target,
            label_priors=label_priors
        )

        base_utils.validate_lambda_term(lambda_term, lambdas)

        # positive(negative)_weights' shape = [C, K] or [C]
        positive_weights, negative_weights = self.get_positive_negative_weights(
            lambdas=self.lambdas,
            targets=self.at_target_values if self.is_auc else self.at_target
        )

        if self.is_auc:

            positive_weights, negative_weights = \
                base_utils.validate_prepare_positive_negative_weights(
                    positive_weights=positive_weights,
                    negative_weights=negative_weights,
                    required_shape=(self.num_labels, self.num_anchors),
                    device=logits.device
                )
            # hinge_loss's shape = [N,C,K]
            cross_entropy_loss = base_utils.weighted_cross_entropy_loss(
                labels.unsqueeze(-1),
                logits.unsqueeze(-1) - self.biases,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
            )

            per_anchor_loss = weights.unsqueeze(-1) * cross_entropy_loss - lambda_term
            # loss = per_label_loss
            loss = per_anchor_loss.sum(2) * self.delta
            loss /= (self.at_target_range[1] - self.at_target_range[0] - self.delta)

        else:

            positive_weights, negative_weights = \
                base_utils.validate_prepare_positive_negative_weights(
                    positive_weights=positive_weights,
                    negative_weights=negative_weights,
                    required_shape=(self.num_labels,),
                    device=logits.device
                )

            # hinge_loss's shape = [N,C]
            cross_entropy_loss = base_utils.weighted_cross_entropy_loss(
                labels,
                logits - self.biases,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
            )
            loss = weights * cross_entropy_loss - lambda_term

        if not reduce:
            return loss
        elif size_average:
            return loss.mean()
        else:
            return loss.sum()

    @staticmethod
    def get_lambda_term(lambdas, targets, label_priors):
        raise NotImplementedError

    @staticmethod
    def get_positive_negative_weights(lambdas, targets):
        raise NotImplementedError






