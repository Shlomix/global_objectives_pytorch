import torch.nn as nn
import utils_new as utils
from utils_new import FloatTensor


class BaseLoss(nn.Module):

    def __init__(self,
                 num_classes=1,
                 auc=False, num_anchors=1):

        nn.Module.__init__(self)

        self.num_classes = num_classes
        self.auc = auc
        self.num_anchors = num_anchors

        self.biases = nn.Parameter(
            FloatTensor(self.num_classes, self.num_anchors).zero_()
        )
        self.lambdas = nn.Parameter(
            FloatTensor(self.num_classes, self.num_anchors).data.fill_(
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

        # class_priors' shape = [num_classes]
        class_priors = utils.calc_class_priors(labels, weights=weights)
        # lambdas' shape = [num_classes, num_anchors]
        lambdas = utils.lagrange_multiplier(self.lambdas)
        positive_weights, negative_weights = self.calc_pos_neg_weights(lambdas)

        lambda_term = self.calc_lambda_term(lambdas, class_priors)

        hinge_loss = utils.calc_hinge_loss(
            labels.unsqueeze(-1),
            logits.unsqueeze(-1),
            positive_weights=positive_weights,
            negative_weights=negative_weights,
        )

        per_anchor_loss = weights.unsqueeze(-1) * hinge_loss + lambda_term

        if self.auc:
            loss = per_anchor_loss.sum(2) * self.delta
            loss /= self.fp_range[1] - self.fp_range[0]
        else:
            loss = per_anchor_loss.sum(2)

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


