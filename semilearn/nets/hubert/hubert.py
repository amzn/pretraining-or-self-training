# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0


import torch
import torch.nn as nn
from transformers import HubertModel


class ClassificationHubert(nn.Module):
    """
    A PyTorch model for text classification using HuBERT.

    Args:
        name: The name of the pre-trained HuBERT model to load.
        num_classes: The number of classes to classify.
    """
    def __init__(self, name, num_classes=2):
        super(ClassificationHubert, self).__init__()
        self.model = HubertModel.from_pretrained(name)
        # for vat
        self.model.feature_extractor._requires_grad = False 
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, num_classes)
        ])

    def forward(self, x, only_fc=False, only_feat=False, **kwargs):
        """
        Forward pass through the model.

        Args:
            x: Input tensor, a batch of audio sequences.
            only_fc: Whether to use only the classification head.
            only_feat: Whether to return only the pooled features.

        Returns:
            A dictionary containing the logits and/or pooled features, depending on the
            flags passed.
        """
        if only_fc:
            logits = self.classifier(x)
            return logits

        pooled_output = self.extract(x)

        if only_feat:
            return pooled_output

        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}
        return result_dict

    def extract(self, x):
        """
        Extract the pooled features from a batch of audio sequences.

        Args:
            x : Input tensor, a batch of audio sequences.

        Returns:
            A tensor containing the pooled features for each audio sequence.
        """

        out_dict = self.model(x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        embed = out_dict['hidden_states'][0]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        """
        Get a dictionary mapping group names to regular expressions matching the
        parameters in those groups.

        Args:
            coarse: Whether to match the parameters in a coarse-grained way.
            prefix: A prefix to add to all of the regular expressions.

        Returns:
            A dictionary mapping group names to regular expressions matching the
            parameters in those groups.
        """
        matcher = dict(stem=r'^{}model.feature_projection|^{}model.feature_extractor|^{}model.encoder.pos_conv_embed'.format(prefix, prefix, prefix), blocks=r'^{}model.encoder.layers.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        """
        Get a list of parameter names that should not be decayed when training the model.

        Returns:
            A list of parameter names that should not be decayed when training the model.
        """
        return []


def hubert_base(pretrained=False, pretrained_path=None, **kwargs):
    model = ClassificationHubert(name='facebook/hubert-base-ls960', **kwargs)
    return model


if __name__ == '__main__':
    model = hubert_base()
    print(model)
