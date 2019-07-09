import torch.nn as nn
from base import BaseLoss
import utils_new as utils


class AUCROCLoss(BaseLoss):

    def __init__(self, fp_range_lower=0.0, fp_range_upper=1.0,
                 num_classes=1, num_anchors=20):
        nn.Module.__init__(self)
        super(AUCROCLoss, self).__init__(
            auc=True,
            target_type="fp",
            target=(fp_range_lower, fp_range_upper),
            num_classes=num_classes,
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
                 num_classes=1, num_anchors=20):
        nn.Module.__init__(self)
        super(AUCPRLoss, self).__init__(
            auc=True,
            target_type="precision",
            target=(precision_range_lower, precision_range_upper),
            num_classes=num_classes,
            num_anchors=num_anchors
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.target_values)
        )
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = 1.0 + lambdas * (1.0 - self.target_values)
        neg_weight = lambdas * self.target_values
        return pos_weight, neg_weight


class PRLoss(BaseLoss):

    def __init__(self, target_recall, num_classes=1):
        nn.Module.__init__(self)
        super(PRLoss, self).__init__(
            auc=False,
            target_type="recall",
            target=target_recall,
            num_classes=num_classes,
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = class_priors * (
            lambdas * (self.target - 1.0)
        )
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = lambdas
        neg_weight = 1.0
        return pos_weight, neg_weight


class TPRFPRLoss(BaseLoss):

    def __init__(self, target_fpr, num_classes=1):
        nn.Module.__init__(self)
        super(TPRFPRLoss, self).__init__(
            auc=True,
            target_type="fpr",
            target=target_fpr,
            num_classes=num_classes,
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


