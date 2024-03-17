# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from PIL import Image

from semilearn.core import AlgorithmBase
from semilearn.core.utils import get_data_loader
from semilearn.algorithms.hooks import FixedThresholdingHook
from semilearn.algorithms.utils import ce_loss, SSL_Argument, str2bool


def rotate_img(img, rot):
    if rot == 0:  # 0 degrees rotation
        return img
    elif rot == 90:  # 90 degrees rotation
        return img.rot90(1, [1, 2])
    elif rot == 180:  # 90 degrees rotation
        return img.rot90(2, [1, 2])
    elif rot == 270:  # 270 degrees rotation / or -90
        return img.rot90(1, [2, 1])
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')


class RotNet(torch.utils.data.Dataset):
    """
    Dataloader for RotNet.

    The image first goes through data augmentation, and then it is rotated 4 times.
    The output is 4 rotated views of the augmented image, and the corresponding labels are 0, 1, 2, 3.

    Args:
        data: The dataset containing image data.
        transform (callable, optional): A function/transform to apply to the image data.
        target_transform (callable, optional): A function/transform to apply to the target (labels).

    """
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Get an item from the RotNet dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - img (PIL.Image.Image): The augmented image.
                - rotated_img_90 (PIL.Image.Image): The image rotated 90 degrees.
                - rotated_img_180 (PIL.Image.Image): The image rotated 180 degrees.
                - rotated_img_270 (PIL.Image.Image): The image rotated 270 degrees.
                - rotation_labels (torch.Tensor): The labels corresponding to rotations (0, 1, 2, 3).

        """
        img = self.data[index]

        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        elif isinstance(img, str):
            img = Image.open(img)
        img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        rotation_labels = torch.LongTensor([0, 1, 2, 3])
        return img, rotate_img(img, 90), rotate_img(img, 180), rotate_img(img, 270), rotation_labels

    def __len__(self):
        """
        Get the length of the RotNet dataset.
        """
        return len(self.data)


class CRMatch_Net(nn.Module):
    """
    Contrastive and Rotation Matching Network.

    Args:
        base (nn.Module): The base neural network architecture.
        args: Configuration arguments for the network.
        use_rot (bool, optional): Whether to use rotation matching. Default is True.

    Attributes:
        backbone (nn.Module): The base neural network.
        use_rot (bool): Whether to use rotation matching.
        feat_planes (int): The number of feature planes in the base network.
        args: Configuration arguments for the network.
        rot_classifier (nn.Sequential): The rotation classifier (if using rotation matching).
        ds_classifier (nn.Linear): The downstream classifier.

    """
    def __init__(self, base, args, use_rot=True):
        super(CRMatch_Net, self).__init__()
        self.backbone = base
        self.use_rot = use_rot
        self.feat_planes = base.num_features
        self.args = args

        if self.use_rot:
            self.rot_classifier = nn.Sequential(
                nn.Linear(self.feat_planes, self.feat_planes),
                nn.ReLU(inplace=False),
                nn.Linear(self.feat_planes, 4)
            )
        if 'wrn' in args.net or 'resnet' in args.net:
            if args.dataset == 'stl10':
                feat_map_size = 6 * 6 * self.feat_planes
            elif args.dataset == 'imagenet':
                feat_map_size = 7 * 7 * self.feat_planes
            else:
                feat_map_size = 8 * 8 * self.feat_planes
        elif 'vit' in args.net or 'bert' in args.net or 'wave2vec' in args.net:
            feat_map_size = self.backbone.num_features
        else:
            raise NotImplementedError
        self.ds_classifier = nn.Linear(feat_map_size, self.feat_planes, bias=True)

    def forward(self, x):
        """
        Forward pass through the CRMatch_Net network.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            dict: Dictionary containing various logits and feature tensors.
                - 'logits' (`torch.Tensor`): Logits from the main classification head.
                - 'logits_ds' (`torch.Tensor`): Logits from the downstream classifier.
                - 'feat' (`torch.Tensor`): Feature maps extracted from the input.
                - 'logits_rot' (`torch.Tensor`, optional): Logits from the rotation classifier (if enabled).

        """
        feat_maps = self.backbone.extract(x)

        if 'wrn' in self.args.net or 'resnet' in self.args.net:
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
            feat_maps = torch.mean(feat_maps, dim=(2, 3))
        elif 'vit' in self.args.net:
            if self.backbone.global_pool:
                feat_maps = feat_maps[:, 1:].mean(dim=1) if self.backbone.global_pool == 'avg' else feat_maps[:, 0]
            feat_maps = self.backbone.fc_norm(feat_maps)
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
        elif 'bert' in self.args.net or 'wave2vec' in self.args.net:
            logits_ds = self.ds_classifier(feat_maps.view(feat_maps.size(0), -1))
        else:
            raise NotImplementedError
        logits = self.backbone(feat_maps, only_fc=True)
        results_dict = {'logits':logits, 'logits_ds':logits_ds, 'feat':feat_maps}
        # feat_flat = torch.mean(feat_maps, dim=(2, 3))
        # logits = self.backbone(feat_flat, only_fc=True)
        if self.use_rot:
            logits_rot = self.rot_classifier(feat_maps)
            results_dict['logits_rot'] = logits_rot
        else:
            results_dict['logits_rot'] = None
        return results_dict

    def group_matcher(self, coarse=False):
        """
        Get the group matcher from the backbone network.

        Args:
            coarse (bool, optional): Whether to use a coarse matcher. Default is False.

        Returns:
            nn.Module: The group matcher module.

        """
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher


class CRMatch(AlgorithmBase):
    """
        CRMatch algorithm (https://arxiv.org/abs/2112.05825).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
        """
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        self.lambda_rot = args.rot_loss_ratio
        self.use_rot = self.lambda_rot > 0
        super().__init__(args, net_builder,  tb_log, logger)
        # crmatch specificed arguments
        self.init(p_cutoff=args.p_cutoff, hard_label=args.hard_label)
        

    def init(self, p_cutoff, hard_label=True):
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label

    def set_data_loader(self):
        """
        Set data loaders for training.

        Returns:
            dict: Dictionary containing data loaders.

        """
        loader_dict = super().set_data_loader()

        if self.use_rot:
            x_ulb_rot = deepcopy(loader_dict['train_ulb'].dataset.data)
            dataset_ulb_rot = RotNet(x_ulb_rot, transform=loader_dict['train_lb'].dataset.transform)
            loader_dict['train_ulb_rot'] = get_data_loader(self.args,
                                                           dataset_ulb_rot,
                                                           self.args.batch_size,
                                                           data_sampler=self.args.train_sampler,
                                                           num_iters=self.num_train_iter,
                                                           num_epochs=self.epochs,
                                                           num_workers=4 * self.args.num_workers,
                                                           distributed=self.distributed)
            loader_dict['train_ulb_rot_iter'] = iter(loader_dict['train_ulb_rot'])
        return loader_dict

    def set_model(self):
        model = super().set_model()
        model = CRMatch_Net(model, self.args, use_rot=self.use_rot)
        return model
    
    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = CRMatch_Net(ema_model, self.args, use_rot=self.use_rot)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model

    def set_hooks(self):
        """
        Set hooks for the CRMatch algorithm.

        """
        self.register_hook(FixedThresholdingHook(), "MaskingHook")
        super().set_hooks()


    def train(self):
        
        self.model.train()
        self.call_hook("before_run")

        for epoch in range(self.epochs):
            self.epoch = epoch
            
            # prevent the training iterations exceed args.num_train_iter
            if self.it > self.num_train_iter:
                break
                
            self.call_hook("before_train_epoch")
            
            for data_lb, data_ulb in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
                # prevent the training iterations exceed args.num_train_iter
                if self.it > self.num_train_iter:
                    break

                self.call_hook("before_train_step")

                if self.use_rot:
                    try:
                        img, img90, img180, img270, rot_v = next(self.loader_dict['train_ulb_rot_iter'])
                    except:
                        self.loader_dict['train_ulb_rot_iter'] = iter(self.loader_dict['train_ulb_rot'])
                        img, img90, img180, img270, rot_v = next(self.loader_dict['train_ulb_rot_iter'])
                    x_ulb_rot = torch.cat((img, img90, img180, img270), dim=0).contiguous()
                    rot_v = rot_v.transpose(1, 0).contiguous().view(-1)
                else:
                    x_ulb_rot = None
                    rot_v = None

                self.tb_dict = self.train_step(**self.process_batch(**data_lb, **data_ulb, x_ulb_rot=x_ulb_rot, rot_v=rot_v))

                self.call_hook("after_train_step")
                self.it += 1

            self.call_hook("after_train_epoch")

        self.call_hook("after_run")


    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, x_ulb_rot=None, rot_v=None):
        """
        Perform a single training step.

        Args:
            x_lb (torch.Tensor): Labeled input data.
            y_lb (torch.Tensor): Labeled targets.
            x_ulb_w (torch.Tensor or dict): Unlabeled input data (weakly augmented).
            x_ulb_s (torch.Tensor): Unlabeled input data (strongly augmented).
            x_ulb_rot (torch.Tensor, optional): Unlabeled input data for rotation matching (if using rotation matching).
            rot_v (torch.Tensor, optional): Rotation labels for rotation matching (if using rotation matching).

        Returns:
            dict: Dictionary containing training metrics.

        """
        num_lb = y_lb.shape[0]
        num_ulb = len(x_ulb_w['input_ids']) if isinstance(x_ulb_w, dict) else x_ulb_w.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                if self.use_rot:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s, x_ulb_rot), dim=0).contiguous()
                else:
                    inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s), dim=0).contiguous()
                outputs = self.model(inputs)
                logits, logits_rot, logits_ds = outputs['logits'], outputs['logits_rot'], outputs['logits_ds']
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:num_lb + 2 * num_ulb].chunk(2)
                logits_ds_w, logits_ds_s = logits_ds[num_lb:num_lb + 2 * num_ulb].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                # logits_x_lb, _, _ = self.model(x_lb)
                
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s,logits_ds_s = outs_x_ulb_s['logits'], outs_x_ulb_s['logits_ds']
                # logits_x_ulb_s, _, logits_ds_s = self.model(x_ulb_s)
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w, logits_ds_w = outs_x_ulb_w['logits'], outs_x_ulb_w['logits_ds']
            
            with torch.no_grad():
                y_ulb = torch.argmax(logits_x_ulb_w, dim=-1)
                mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w)    

            Lx = ce_loss(logits_x_lb, y_lb, reduction='mean')
            Lu = (ce_loss(logits_x_ulb_s, y_ulb, reduction='none') * mask).mean()
            Ld = F.cosine_embedding_loss(logits_ds_s, logits_ds_w, -torch.ones(logits_ds_s.size(0)).float().cuda(self.gpu), reduction='none')
            Ld = (Ld * mask).mean()

            total_loss = Lx + Lu + Ld

            if self.use_rot:
                if self.use_cat:
                    logits_rot = logits_rot[num_lb + 2 * num_ulb:]
                else:
                    logits_rot = self.model(x_ulb_rot)['logits_rot']
                Lrot = ce_loss(logits_rot, rot_v, reduction='mean')
                total_loss += Lrot

        # parameter updates
        self.call_hook("param_update", "ParamUpdateHook", loss=total_loss)

        tb_dict = {}
        tb_dict['train/sup_loss'] = Lx.item()
        tb_dict['train/unsup_loss'] = Lu.item()
        tb_dict['train/total_loss'] = total_loss.item()
        tb_dict['train/mask_ratio'] = mask.float().mean().item()
        return tb_dict


    @staticmethod
    def get_argument():
        """
        Get algorithm-specific arguments.
        """
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--rot_loss_ratio', float, 1.0, 'weight for rot loss, set to 0 for nlp and speech'),
            SSL_Argument('--p_cutoff', float, 0.95),
        ]
