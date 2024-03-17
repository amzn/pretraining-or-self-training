# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class ClassificationWave2Vec(nn.Module):
    """Classification model using Wav2Vec2 as a feature extractor.

        Args:
            name (str): Pretrained model name or path.
            num_classes (int): Number of classes for classification.

    """
    def __init__(self, name, num_classes=2):
        super(ClassificationWave2Vec, self).__init__()
        self.model = Wav2Vec2Model.from_pretrained(name)
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
            x: Input tensor, depends on only_fc and only_feat flag.
            only_fc (bool): Only use classifier, input should be features before classifier.
            only_feat (bool): Only return pooled features.

        Returns:
            torch.Tensor or dict: If only_fc is True, returns classifier output.
                                   If only_feat is True, returns pooled features.
                                   Otherwise, returns a dictionary with 'logits' for classifier output and 'feat' for pooled features.
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
        """Extract features from the input.

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: Pooled features.

        """
        out_dict = self.model(x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        embed = out_dict['hidden_states'][0]
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        """Define a group matcher for layer-wise weight decay.

        Args:
            coarse (bool): If True, use a coarse matcher.
            prefix (str): Prefix for layer names.

        Returns:
            dict: Matcher dictionary.

        """
        matcher = dict(stem=r'^{}model.feature_projection|^{}model.feature_extractor'.format(prefix, prefix), blocks=r'^{}model.encoder.layers.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []

def wave2vecv2_base(pretrained=False,pretrained_path=None, **kwargs):
    """Create a base Wave2Vec2 classification model.

    Args:
        pretrained (bool): Whether to load pretrained weights.
        pretrained_path (str): Path to pretrained weights.
        **kwargs: Additional keyword arguments for the ClassificationWave2Vec constructor.

    Returns:
        ClassificationWave2Vec: Base Wave2Vec2 classification model.

    """
    model = ClassificationWave2Vec(name='facebook/wav2vec2-base-960h', **kwargs)
    return model


if __name__ == '__main__':
    model = wave2vecv2_base()
    print(model)
