# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


from .utils import AdaMatchThresholdingHook
from semilearn.core import AlgorithmBase
from semilearn.algorithms.hooks import PseudoLabelingHook, DistAlignEMAHook
from semilearn.algorithms.utils import ce_loss, consistency_loss,  SSL_Argument, str2bool


class AdaMatch(AlgorithmBase):
    """
    AdaMatch Algorithm for Semi-Supervised Learning.

    Args:
        args: Command-line arguments and configurations.
        net_builder: The network builder for constructing the model.
        tb_log: TensorBoard logging instance (optional).
        logger: Logging instance (optional).

    Attributes:
        p_cutoff (float): The confidence threshold for masking.
        T (float): Temperature parameter for pseudo-label generation.
        use_hard_label (bool): Whether to use hard labels for pseudo-labeling.
        dist_align (bool): Whether to perform distribution alignment.
        ema_p (float): Exponential moving average parameter.

    """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(p_cutoff=args.p_cutoff, T=args.T, hard_label=args.hard_label, dist_align=args.dist_align, ema_p=args.ema_p)
    
    def init(self, p_cutoff, T, hard_label=True, dist_align=True, ema_p=0.999):
        """
        Initialize the AdaMatch algorithm parameters.

       """
        self.p_cutoff = p_cutoff
        self.T = T
        self.use_hard_label = hard_label
        self.dist_align = dist_align
        self.ema_p = ema_p


    def set_hooks(self):
        """
        Set up hooks for the AdaMatch algorithm.
        """
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(
            DistAlignEMAHook(num_classes=self.num_classes, momentum=self.args.ema_p, p_target_type='model'), 
            "DistAlignHook")
        self.register_hook(AdaMatchThresholdingHook(), "MaskingHook")
        super().set_hooks()


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
        """
        Perform a single training step for AdaMatch.

        Args:
            x_lb (torch.Tensor): Labeled data inputs.
            y_lb (torch.Tensor): Labeled data ground truth labels.
            x_ulb_w (torch.Tensor): Unlabeled data inputs for weak augmentation.
            x_ulb_s (torch.Tensor): Unlabeled data inputs for strong augmentation.

        Returns:
            dict: A dictionary containing training statistics.

        """
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    

            sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

            probs_x_lb = torch.softmax(logits_x_lb.detach(), dim=-1)
            probs_x_ulb_w = torch.softmax(logits_x_ulb_w.detach(), dim=-1)

            # distribution alignment 
            probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w, probs_x_lb=probs_x_lb)

            # calculate weight
            mask = self.call_hook("masking", "MaskingHook", logits_x_lb=probs_x_lb, logits_x_ulb=probs_x_ulb_w, softmax_x_lb=False, softmax_x_ulb=False)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            # calculate loss
            unsup_loss = consistency_loss(logits_x_ulb_s,
                                          pseudo_label,
                                          'ce',
                                          mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = sup_loss.item()
        tb_dict['train/unsup_loss'] = unsup_loss.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.mean().item()
        return tb_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['DistAlignHook'].p_model.cpu()
        save_dict['p_target'] = self.hooks_dict['DistAlignHook'].p_target.cpu()
        return save_dict


    def load_model(self, load_path):
        """
        Load a trained model from a checkpoint file.

        Args:
            load_path (str): Path to the checkpoint file.

        Returns:
            dict: A dictionary containing the loaded model and additional parameters.

        """
        checkpoint = super().load_model(load_path)
        self.hooks_dict['DistAlignHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['DistAlignHook'].p_target = checkpoint['p_target'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        """
        Get a list of arguments and their types for configuring AdaMatch.

        Returns:
            List[SSL_Argument]: A list of arguments and their types.
        """
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--dist_align', str2bool, True),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
