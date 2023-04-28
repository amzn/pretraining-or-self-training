# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# Ref: https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/sampler_seed.py

from torch.utils.data import DataLoader
from .hook import Hook
from semilearn.datasets import DistributedSampler

class DistSamplerSeedHook(Hook):
    def __init__(self) -> None:
        super().__init__()
    
    def before_train_epoch(self, algorithm):
        for name, dataloader in algorithm.loader_dict.items():
            if not isinstance(dataloader, DataLoader):
                continue

            if isinstance(dataloader.sampler, DistributedSampler):
                algorithm.loader_dict[name].sampler.set_epoch(algorithm.epoch)
