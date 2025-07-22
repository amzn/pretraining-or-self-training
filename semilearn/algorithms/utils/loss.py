# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch 
import torch.nn as nn 
from torch.nn import functional as F


def smooth_targets(logits, targets, smoothing=0.1):
    """
    Apply label smoothing to the target labels.

    Args:
        logits (torch.Tensor): Logit values, shape=[Batch size, # of classes].
        targets (torch.Tensor): Integer or vector representing the target labels, shape=[Batch size] or [Batch size, # of classes].
        smoothing (float): Smoothing factor for label smoothing.

    Returns:
        torch.Tensor: Smoothed target distribution.

    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
    return true_dist


def ce_loss(logits, targets, reduction='none'):
    """
    Compute cross-entropy loss.

    Args:
        logits (torch.Tensor): Logit values, shape=[Batch size, # of classes].
        targets (torch.Tensor): Integer or vector representing the target labels, shape=[Batch size] or [Batch size, # of classes].
        reduction (str): The reduction argument for the loss. Options: 'none', 'mean', 'sum'.

    Returns:
        torch.Tensor: Cross-entropy loss.

    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    Compute consistency regularization loss in semi-supervised learning.

    Args:
        logits (torch.Tensor): Logits used to calculate the loss and back-propagate. Typically, strong-augmented unlabeled samples.
        targets (torch.Tensor): Pseudo-labels, either hard label or soft label.
        name (str): Type of loss to calculate. Options: 'ce' (cross-entropy), 'mse' (mean squared error).
        mask (torch.Tensor, optional): Mask to mask out samples when calculating the loss. Used for confidence masking.

    Returns:
        torch.Tensor: Consistency regularization loss.

    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()
