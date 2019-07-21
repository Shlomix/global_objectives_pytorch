import torch
import torch.nn as nn
import global_objectives.base_utils as utils

class LagrangeMultiplier(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, dual_factor):
        ctx.save_for_backward(input)
        ctx.dual_factor = dual_factor
        return torch.abs(input)

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.dual_factor * grad_output.neg(), \
               None


def lagrange_multiplier(x, dual_factor=1.0):
    return LagrangeMultiplier.apply(x, dual_factor)


class BaseLoss(nn.Module):

    def __init__(self, at_target, at_target_type, num_labels,
                 dual_factor=0.1, loss_type='cross_entropy',
                 is_auc=False, num_anchors=20):
        nn.Module.__init__(self)

        utils.validate_init_args(
            at_target=at_target,
            at_target_type=at_target_type,
            num_labels=num_labels,
            dual_factor=dual_factor,
            loss_type=loss_type,
            is_auc=is_auc,
            num_anchors=num_anchors
        )

        self.num_labels = num_labels
        self.dual_factor = dual_factor
        self.at_target_type = at_target_type
        self.loss_type = loss_type
        self.is_auc = is_auc

        if loss_type == 'cross_entropy':
            self.loss_func = utils.weighted_cross_entropy_loss
        else:
            self.loss_func = utils.weighted_hinge_loss

        if self.is_auc:
            self.required_shape = (self.num_labels, self.num_anchors)
            self.num_anchors = num_anchors
            self.at_target_range = at_target

            _at_target_values, self.delta = utils.build_anchors(
                self.at_target_range, self.num_anchors
            )
            self.register_buffer('at_target_values', _at_target_values)
        else:
            self.required_shape = (self.num_labels,)
            self.register_buffer('at_target', torch.tensor(at_target))

        self.biases = nn.Parameter(
            torch.FloatTensor(self.required_shape).zero_()
        )
        self.lambdas = nn.Parameter(
            torch.FloatTensor(self.required_shape).data.fill_(
                1.0
            )
        )

    def forward(self, logits, labels,
                class_priors=None, weights=None,
                reduce=True, size_average=True):

        logits, labels, weights = \
            utils.validate_prepare_logits_labels_weights(
                logits=logits, labels=labels,
                weights=weights, num_labels=self.num_labels
        )

        label_counts, _ = utils.build_label_counts_priors(
            labels=labels, weights=weights
        )

        lambdas = lagrange_multiplier(
            x=self.lambdas, dual_factor=self.dual_factor
        )

        if self.is_auc:

            lambda_term = self.get_lambda_term(
                lambdas=lambdas,
                targets=self.at_target_values,
                label_priors=label_counts
            )

            positive_weights, negative_weights = self.get_positive_negative_weights(
                lambdas=lambdas,
                targets=self.at_target_values,
            )

            positive_weights, negative_weights = \
                utils.prepare_positive_negative_weights(
                    positive_weights=positive_weights,
                    negative_weights=negative_weights,
                    required_shape=(self.num_labels, self.num_anchors),
                    device=logits.device
                )
            # hinge_loss's shape = [N,C,K]
            weighted_loss = self.loss_func(
                labels.unsqueeze(-1),
                logits.unsqueeze(-1) - self.biases,
                positive_weights=positive_weights,
                negative_weights=negative_weights,
            )

            per_anchor_loss = weights.unsqueeze(-1) * weighted_loss - lambda_term
            # loss = per_label_loss
            loss = per_anchor_loss.sum(2) * self.delta
            loss /= (self.at_target_range[1] - self.at_target_range[0] - self.delta)

        else:

            lambda_term = self.get_lambda_term(
                lambdas=lambdas,
                targets=self.at_target,
                label_priors=label_counts
            )

            positive_weights, negative_weights = self.get_positive_negative_weights(
                lambdas=lambdas,
                targets=self.at_target,
            )

            positive_weights, negative_weights = \
                utils.prepare_positive_negative_weights(
                    positive_weights=positive_weights,
                    negative_weights=negative_weights,
                    required_shape=(self.num_labels,),
                    device=logits.device
                )

            # hinge_loss's shape = [N,C]
            cross_entropy_loss = self.loss_func(
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



