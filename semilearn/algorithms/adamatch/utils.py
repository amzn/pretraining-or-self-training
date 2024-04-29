# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from semilearn.algorithms.hooks import MaskingHook

class AdaMatchThresholdingHook(MaskingHook):
    """
    Relative Confidence Thresholding in AdaMatch.

    This class implements relative confidence thresholding as a masking strategy in the AdaMatch algorithm.

    Args:
        algorithm (AdaMatch): The AdaMatch algorithm instance.
        logits_x_lb (torch.Tensor): Logits for labeled data.
        logits_x_ulb (torch.Tensor): Logits for unlabeled data.
        softmax_x_lb (bool, optional): Whether logits_x_lb should be softmaxed. Default is True.
        softmax_x_ulb (bool, optional): Whether logits_x_ulb should be softmaxed. Default is True.

    Returns:
        torch.Tensor: A binary mask indicating which samples pass the confidence threshold.

    Notes:
        This hook calculates the maximum probabilities for labeled and unlabeled data separately
        and applies a confidence threshold based on the mean maximum probability multiplied by `algorithm.p_cutoff`.
    """

    @torch.no_grad()
    def masking(self, algorithm, logits_x_lb, logits_x_ulb, softmax_x_lb=True, softmax_x_ulb=True,  *args, **kwargs):
        """
        Apply relative confidence thresholding to mask data samples.

        Args:
            algorithm (AdaMatch): The AdaMatch algorithm instance.
            logits_x_lb (torch.Tensor): Logits for labeled data.
            logits_x_ulb (torch.Tensor): Logits for unlabeled data.
            softmax_x_lb (bool, optional): Whether logits_x_lb should be softmaxed. Default is True.
            softmax_x_ulb (bool, optional): Whether logits_x_ulb should be softmaxed. Default is True.

        Returns:
            torch.Tensor: A binary mask indicating which samples pass the confidence threshold.
        """
        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        if softmax_x_lb:
            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_lb = logits_x_lb.detach()

        max_probs, _ = probs_x_lb.max(dim=-1)
        p_cutoff = max_probs.mean() * algorithm.p_cutoff
        max_probs, _ = probs_x_ulb.max(dim=-1)
        mask = max_probs.ge(p_cutoff).to(max_probs.dtype)
        return mask
