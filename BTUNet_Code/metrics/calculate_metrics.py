import sys

sys.path.append('../')
from metrics.binary_confusion_matrix import get_binary_confusion_matrix, get_threshold_binary_confusion_matrix
from metrics.binary_statistical_metrics import get_accuracy, get_true_positive_rate, get_true_negative_rate, \
    get_precision, get_f1_socre, get_iou
from metrics.dice_coefficient import hard_dice
from metrics.pr_curve import get_pr_curve
from metrics.roc_curve import get_auroc, get_roc_curve
from util.numpy_utils import tensor2numpy
import numpy as np
from copy import deepcopy

import torch
from torch import nn


def hard_dice(input_, target, threshold=0.5, reduction='mean', epsilon=1e-8):
    """
    Hard dice score coefficient after thresholding.

    Arguments:
        preds (torch tensor): raw probability outputs
        targets (torch tensor): ground truth
        threshold (float): threshold value, default: 0.5
        reduction (string): one of 'none', 'mean' or 'sum'
        epsilon (float): epsilon for numerical stability, default: 1e-8

    Returns:
        dice (torch tensor): hard dice score coefficient
    """
    if not input_.shape == target.shape:
        raise ValueError

    # if not (input_.max() <= 1.0 and input_.min() >= 0.0):
    #     raise ValueError

    if not ((target.max() == 1.0 and target.min() == 0.0 and(target.unique().numel() == 2)) 
        or (target.max() == 0.0 and target.min() == 0.0 and(target.unique().numel() == 1))):
        raise ValueError

    input_threshed = input_.clone()
    # input_threshed[input_ < threshold] = 0.0
    # input_threshed[input_ >= threshold] = 1.0

    intesection = torch.sum(input_threshed * target, dim=-1)
    input_norm = torch.sum(input_threshed, dim=-1)
    target_norm = torch.sum(target, dim=-1)
    dice = torch.div(2.0 * intesection + epsilon,
                     input_norm + target_norm + epsilon)

    if reduction == 'none':
        pass
    elif reduction == 'mean':
        dice = torch.mean(dice)
    elif reduction == 'sum':
        dice = torch.sum(dice)
    else:
        raise NotImplementedError

    return dice


class HardDice(nn.Module):
    """
    Hard dice module
    """

    def __init__(self, threshold=0.5, reduction='mean'):
        super(HardDice, self).__init__()
        self.threshold = threshold
        self.reduction = reduction

    def forward(self, input_, target):
        dice = hard_dice(input_=input_, target=target,
                         threshold=self.threshold,
                         reduction=self.reduction,
                         epsilon=1e-8)
        return dice

# np.seterr(divide='ignore', invalid='ignore')


def calculate_metrics( preds, targets, device,config=None):
    curr_TP, curr_FP, curr_TN, curr_FN = get_binary_confusion_matrix(
        input_=preds, target=targets, device=device, pixel=None,
        threshold=0.5,
        reduction='sum')

    curr_acc = get_accuracy(true_positive=curr_TP,
                            false_positive=curr_FP,
                            true_negative=curr_TN,
                            false_negative=curr_FN)

    curr_recall = get_true_positive_rate(true_positive=curr_TP,
                                         false_negative=curr_FN)

    curr_specificity = get_true_negative_rate(false_positive=curr_FP,
                                              true_negative=curr_TN)

    curr_precision = get_precision(true_positive=curr_TP,
                                   false_positive=curr_FP)

    curr_f1_score = get_f1_socre(true_positive=curr_TP,
                                 false_positive=curr_FP,
                                 false_negative=curr_FN)

    curr_iou = get_iou(true_positive=curr_TP,
                       false_positive=curr_FP,
                       false_negative=curr_FN)

    curr_auroc = get_auroc(preds, targets)

    get_dice = HardDice()
    curr_dice = get_dice(preds, targets)

    return (curr_acc, curr_recall, curr_specificity, curr_precision,
            curr_f1_score, curr_dice, curr_iou, curr_auroc)

