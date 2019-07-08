import torch.nn as nn
from base import BaseLoss
import utils_new as utils


class AUCROCLoss(BaseLoss):

    def __init__(self, fp_range_lower=0.0, fp_range_upper=1.0,
                 num_classes=1, num_anchors=20):
        nn.Module.__init__(self)
        super(AUCROCLoss, self).__init__(
            auc=True, num_classes=num_classes,
            num_anchors=num_anchors
        )

        self.fp_range = (
            fp_range_lower,
            fp_range_upper,
        )

        self.fp_values, self.delta = utils.build_anchors(
            self.fp_range, self.num_anchors
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = (1.0 - class_priors).unsqueeze(-1) * (
                lambdas * self.fp_values
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
            auc=True, num_classes=num_classes,
            num_anchors=num_anchors
        )

        self.fp_range = (
            precision_range_lower,
            precision_range_upper,
        )

        self.precision_values, self.delta = utils.build_anchors(
            self.fp_range, self.num_anchors
        )

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = class_priors.unsqueeze(-1) * (
            lambdas * (1.0 - self.precision_values)
        )
        return lambda_term

    def calc_pos_neg_weights(self, lambdas):
        pos_weight = 1.0 + lambdas * (1.0 - self.precision_values)
        neg_weight = lambdas * self.precision_values
        return pos_weight, neg_weight


class PRLoss(BaseLoss):

    def __init__(self, target_recall, num_classes=1):
        nn.Module.__init__(self)
        super(PRLoss, self).__init__(
            auc=False, num_classes=num_classes,
            num_anchors=1
        )
        self.target_recall = target_recall

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = class_priors * (
            lambdas * (self.target_recall - 1.0)
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
            auc=False, num_classes=num_classes,
            num_anchors=1
        )
        self.target_fpr = target_fpr

    def calc_lambda_term(self, lambdas, class_priors):
        lambda_term = (1.0 - class_priors) * (
            lambdas * self.target_fpr
        )
        return lambda_term

    @staticmethod
    def calc_pos_neg_weights(self, lambdas):
        pos_weight = 1.0
        neg_weight = lambdas
        return pos_weight, neg_weight


