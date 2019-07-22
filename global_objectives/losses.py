import torch.nn as nn
from global_objectives.base import BaseLoss


class AUCROCLoss(BaseLoss):

    def __init__(self,
                 fp_range_lower=0.0, fp_range_upper=1.0,
                 num_labels=1, loss_type='cross_entropy',
                 num_anchors=20, dual_factor=0.1):
        nn.Module.__init__(self)
        super(AUCROCLoss, self).__init__(
            is_auc=True,
            at_target_type="fpr",
            at_target=(fp_range_lower, fp_range_upper),
            dual_factor=dual_factor,
            loss_type=loss_type,
            num_labels=num_labels,
            num_anchors=num_anchors
        )

    @staticmethod
    def get_lambda_term(lambdas, targets, label_priors):
        lambda_term = (1.0 - label_priors).unsqueeze(-1) * (
                lambdas * targets
        )
        return lambda_term

    @staticmethod
    def get_positive_negative_weights(lambdas, targets):
        pos_weight = 1.0
        neg_weight = lambdas
        return pos_weight, neg_weight


class AUCPRLoss(BaseLoss):

    def __init__(self, precision_range_lower=0.0, precision_range_upper=1.0,
                 num_labels=1, loss_type='cross_entropy',
                 num_anchors=20, dual_factor=0.1):
        nn.Module.__init__(self)
        super(AUCPRLoss, self).__init__(
            is_auc=True,
            at_target_type="precision",
            at_target=(precision_range_lower, precision_range_upper),
            dual_factor=dual_factor,
            loss_type=loss_type,
            num_labels=num_labels,
            num_anchors=num_anchors
        )

    @staticmethod
    def get_lambda_term(lambdas, targets, label_priors):
        lambda_term = lambdas * (1.0 - targets) * label_priors.unsqueeze(-1)
        return lambda_term

    @staticmethod
    def get_positive_negative_weights(lambdas, targets):
        pos_weight = (1.0 + lambdas) * (1.0 - targets)
        neg_weight = lambdas * targets
        return pos_weight, neg_weight


class PRLoss(BaseLoss):

    def __init__(self, target_recall, num_labels=1, dual_factor=0.1,
                 loss_type='cross_entropy'):
        nn.Module.__init__(self)
        super(PRLoss, self).__init__(
            is_auc=False,
            at_target_type="recall",
            at_target=target_recall,
            dual_factor=dual_factor,
            num_labels=num_labels,
            loss_type=loss_type,
        )

    @staticmethod
    def get_lambda_term(lambdas, targets, label_priors):
        lambda_term = -label_priors * (
            lambdas * (targets - 1.0)
        )
        return lambda_term

    @staticmethod
    def get_positive_negative_weights(lambdas, targets):
        pos_weight = lambdas
        neg_weight = 1.0
        return pos_weight, neg_weight


class TPRFPRLoss(BaseLoss):

    def __init__(self, target_fpr, num_labels=1, dual_factor=0.1,
                 loss_type='cross_entropy'):
        nn.Module.__init__(self)
        super(TPRFPRLoss, self).__init__(
            is_auc=False,
            at_target_type="fpr",
            at_target=target_fpr,
            dual_factor=dual_factor,
            num_labels=num_labels,
            loss_type=loss_type,
        )

    @staticmethod
    def get_lambda_term(lambdas, targets, label_priors):
        lambda_term = (1.0 - label_priors) * (
            lambdas * targets
        )
        return lambda_term

    @staticmethod
    def get_positive_negative_weights(lambdas, targets):
        pos_weight = 1.0
        neg_weight = lambdas
        return pos_weight, neg_weight


