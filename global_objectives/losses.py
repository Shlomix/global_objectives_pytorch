import torch.nn as nn
from global_objectives.base import BaseLoss


class AUCROCLoss(BaseLoss):

    def __init__(self, fpr_range_lower=0.0, fpr_range_upper=1.0,
                 num_labels=1, num_anchors=20, dual_factor=0.1):
        nn.Module.__init__(self)
        super(AUCROCLoss, self).__init__(
            is_auc=True,
            target_type="fpr",
            target=(fpr_range_lower, fpr_range_upper),
            dual_factor=dual_factor,
            num_labels=num_labels,
            num_anchors=num_anchors
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = (1.0 - class_priors).unsqueeze(-1) * (
                lambdas * self.target_values
        )
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = 1.0
        neg_weight = lambdas
        return pos_weight, neg_weight


class AUCPRLoss(BaseLoss):

    def __init__(self, precision_range_lower=0.0, precision_range_upper=1.0,
                 num_labels=1, num_anchors=20, dual_factor=0.1):
        nn.Module.__init__(self)
        super(AUCPRLoss, self).__init__(
            is_auc=True,
            target_type="precision",
            target=(precision_range_lower, precision_range_upper),
            dual_factor=dual_factor,
            num_labels=num_labels,
            num_anchors=num_anchors
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = lambdas * (1.0 - self.target_values) * class_priors.unsqueeze(-1)
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = (1.0 + lambdas) * (1.0 - self.target_values)
        neg_weight = lambdas * self.target_values
        return pos_weight, neg_weight


class PRLoss(BaseLoss):

    def __init__(self, target_recall, num_labels=1, dual_factor=0.1):
        nn.Module.__init__(self)
        super(PRLoss, self).__init__(
            is_auc=False,
            target_type="recall",
            target=target_recall,
            dual_factor=dual_factor,
            num_labels=num_labels,
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = -class_priors * (
            lambdas * (self.target - 1.0)
        )
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = lambdas
        neg_weight = 1.0
        return pos_weight, neg_weight


class TPRFPRLoss(BaseLoss):

    def __init__(self, target_fpr, num_labels=1, dual_factor=0.1):
        nn.Module.__init__(self)
        super(TPRFPRLoss, self).__init__(
            is_auc=False,
            target_type="fpr",
            target=target_fpr,
            dual_factor=dual_factor,
            num_labels=num_labels,
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = (1.0 - class_priors) * (
            lambdas * self.target
        )
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = 1.0
        neg_weight = lambdas
        return pos_weight, neg_weight
