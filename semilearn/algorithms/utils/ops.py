# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np 


def interleave_offsets(batch, nu):
    """
    Compute offsets for interleaving data samples into batches.

    Args:
        batch (int): Total number of data samples.
        nu (int): Number of interleaved datasets.

    Returns:
        list: List of offsets for interleaving.
    """
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    """
    Interleave multiple datasets into batches.

    Args:
        xy (list of lists): List of datasets to interleave.
        batch (int): Batch size.

    Returns:
        list: List of interleaved datasets.
    """
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Perform an all_gather operation on the provided tensors.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Concatenated tensor from all processes.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor)

    output = torch.cat(tensors_gather, dim=0)
    return output


@torch.no_grad()
def mixup_one_target(x, y, alpha=1.0, is_bias=False):
    """
    Apply mixup augmentation to a single target.

    Args:
        x (torch.Tensor): Input data.
        y (torch.Tensor): Target data.
        alpha (float): Mixup hyperparameter.
        is_bias (bool): Whether to use bias when computing lambda.

    Returns:
        torch.Tensor: Mixed input data.
        torch.Tensor: Mixed target data.
        float: Mixup lambda value.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    if is_bias:
        lam = max(lam, 1 - lam)

    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y, lam
