# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from transformers import RobertaModel
import os

class ClassificationRoberta(nn.Module):
    """
    Classification model based on RoBERTa.

    Args:
        name (str): Name of the pretrained RoBERTa model to load.
        num_classes (int, optional): Number of output classes. Default is 2.

    Attributes:
        roberta (RobertaModel): The pretrained RoBERTa model.
        dropout (nn.Dropout): Dropout layer.
        num_features (int): Number of output features from RoBERTa.
        classifier (nn.Sequential): Classifier for final predictions.

    """
    def __init__(self, name, num_classes=2):
        super(ClassificationRoberta, self).__init__()
        # Load pre-trained bert model
        self.roberta = RobertaModel.from_pretrained(name)  # "roberta-base"
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        print(name)
        self.num_features = 1024 if "large" in name else 768
        self.classifier = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.GELU(),
            nn.Linear(self.num_features, num_classes)
        ])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Forward pass for the model.

        Args:
            x (dict): Input tensor, depends on only_fc and only_feat flag.
            only_fc (bool): Only use the classifier, input should be features before classifier.
            only_feat (bool): Only return pooled features.
            return_embed (bool): Return word embeddings, used for VAT.

        Returns:
            dict: Dictionary containing 'logits' for classification logits, 'feat' for pooled features, and 'embed' for word embeddings if return_embed is True.

        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        
        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)

        # Method 1
        pooled_output = self.dropout(out_dict['pooler_output'])

        # Method 2
        # last_hidden = out_dict['last_hidden_state']
        # drop_hidden = self.dropout(last_hidden)
        # pooled_output = torch.mean(drop_hidden, 1)

        if only_feat:
            return pooled_output
        
        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
            
        return result_dict
        
        
    def extract(self, x):
        """
        Extract features from RoBERTa without classification.

        Args:
            x (dict): Input tensor.

        Returns:
            torch.Tensor: Pooled features.

        """
        out_dict = self.roberta(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        return pooled_output

    def group_matcher(self, coarse=False, prefix=''):
        """
        Define a group matcher for layer-wise weight decay.

        Args:
            coarse (bool): If True, use a coarse matcher.
            prefix (str): Prefix for layer names.

        Returns:
            dict: Matcher dictionary.

        """
        matcher = dict(stem=r'^{}roberta.embeddings'.format(prefix), blocks=r'^{}roberta.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []



def roberta_base(pretrained=True, pretrained_path=None, **kwargs):
    """
    Load a pretrained RoBERTa model.

    Args:
        pretrained (bool, optional): Whether to use a pretrained model. Default is True.
        pretrained_path (str, optional): Path to a pretrained model. If not provided, 'roberta-large' is used by default.

    Returns:
        ClassificationRoberta: Pretrained RoBERTa model.

    """
    if not pretrained_path:
        # pretrained_path = 'roberta-base'
        pretrained_path = 'roberta-large'
    print('Loading pretrained model: {}'.format(pretrained_path))
    model = ClassificationRoberta(name=pretrained_path, **kwargs)
    return model
