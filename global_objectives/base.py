import torch
import torch.nn as nn
import global_objectives.utils as utils


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
            self.num_anchors = num_anchors
            self.required_shape = (self.num_labels, self.num_anchors)
            self.at_target_range = at_target

            _at_target_values, self.delta = utils.build_anchors(
                self.at_target_range, self.num_anchors
            )
            self.register_buffer('at_target_values', _at_target_values)
        else:
            self.required_shape = (self.num_labels,)
            self.register_buffer('at_target', torch.tensor(at_target))

        self.biases = nn.Parameter(
            torch.zeros(self.required_shape)
        )
        self.lambdas = nn.Parameter(
            torch.ones(self.required_shape)
        )

    def forward(self, logits, labels,
                label_priors=None, weights=None,
                reduce=True, size_average=True):

        logits, labels, weights = \
            utils.validate_prepare_logits_labels_weights(
                logits=logits, labels=labels,
                weights=weights, num_labels=self.num_labels
        )

        label_priors = utils.build_label_priors(
            labels=labels, weights=weights,
            label_priors=label_priors
        )

        lambdas = lagrange_multiplier(
            x=self.lambdas, dual_factor=self.dual_factor
        )

        if self.is_auc:
            _targets = self.at_target_values
            _labels = labels.unsqueeze(-1)
            _logits = logits.unsqueeze(-1)
            _weights = weights.unsqueeze(-1)

        else:
            _targets = self.at_target
            _labels = labels
            _logits = logits
            _weights = weights

        lambda_term = self.get_lambda_term(
            lambdas=lambdas,
            targets=_targets,
            label_priors=label_priors
        )

        positive_weights, negative_weights = \
            self.get_positive_negative_weights(
                lambdas=lambdas, targets=_targets,
        )

        positive_weights, negative_weights = \
            utils.prepare_positive_negative_weights(
                positive_weights=positive_weights,
                negative_weights=negative_weights,
                required_shape=self.required_shape,
                device=logits.device
            )

        weighted_loss = self.loss_func(
            _labels,
            _logits - self.biases,
            positive_weights=positive_weights,
            negative_weights=negative_weights,
        )

        loss = _weights * weighted_loss - lambda_term

        if self.is_auc:
            loss = loss.sum(2) * self.delta
            loss /= (self.at_target_range[1] - self.at_target_range[0] - self.delta)

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

    @property
    def labmda_parameters(self):
        return [self.lambdas]

    @property
    def bias_parameters(self):
        return [self.biases]
