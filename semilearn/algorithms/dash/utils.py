# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
from semilearn.algorithms.utils import ce_loss
from semilearn.algorithms.hooks import MaskingHook

class DashThresholdingHook(MaskingHook):
    """
    Dynamic Threshold in DASH
    """
    
    def __init__(self, rho_min, gamma, C, *args, **kwargs):
        """
        Initialize the DashThresholdingHook.

        Args:
            rho_min (float): Minimum threshold value for dynamic thresholding.
            gamma (float): A parameter for adjusting the threshold value.
            C (float): A constant factor for threshold calculation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        """
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.C = C
        self.rho_update_cnt = 0
        self.rho_init = None
        self.rho_min = rho_min
        self.rho = None
    
    @torch.no_grad()
    def update(self, algorithm):
        """
        Update the threshold value.

        Args:
            algorithm: The parent algorithm.

        """
        if self.rho_init is None:
            self.rho_init = algorithm.rho_init

        # adjust rho every 10 epochs
        if algorithm.it % (10 * algorithm.num_iter_per_epoch) == 0:
            self.rho = self.C * (self.gamma ** -self.rho_update_cnt) * self.rho_init
            self.rho = max(self.rho, self.rho_min)
            self.rho_update_cnt += 1
        
        # use hard labels if rho reduced 0.05
        if self.rho == self.rho_min:
            algorithm.use_hard_label = True
        else:
            algorithm.use_hard_label = False 
    
    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, *args, **kwargs):    
        """
        Apply dynamic thresholding to generate masks.

        Args:
            algorithm: The parent algorithm.
            logits_x_ulb (torch.Tensor): Logits for unlabeled data.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: A mask tensor.

        """
        self.update(algorithm)

        if algorithm.use_hard_label:
            pseudo_label = torch.argmax(logits_x_ulb, dim=-1).detach()
        else:
            pseudo_label = torch.softmax(logits_x_ulb / algorithm.T, dim=-1).detach()
        loss_w = ce_loss(logits_x_ulb, pseudo_label, reduction='none')
        mask = loss_w.le(self.rho).to(logits_x_ulb.dtype)
        return mask
