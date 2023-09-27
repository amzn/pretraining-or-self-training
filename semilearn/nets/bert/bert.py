# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from transformers import BertModel
import os

class ClassificationBert(nn.Module):
    """
    A PyTorch model for text classification using BERT.

    Args:
        name: The name of the pre-trained BERT model to load.
        num_classes: The number of classes to classify.
    """
    def __init__(self, name, num_classes=2):
        super(ClassificationBert, self).__init__()
        # Load pre-trained bert model
        self.bert = BertModel.from_pretrained(name)
        self.dropout = torch.nn.Dropout(p=0.1, inplace=False)
        self.num_features = 768
        self.classifier = nn.Sequential(*[
            nn.Linear(768, 768),
            nn.GELU(),
            nn.Linear(768, num_classes)
        ])

    def forward(self, x, only_fc=False, only_feat=False, return_embed=False, **kwargs):
        """
        Forward pass through the model.

        Args:
            x: Input tensor, a batch of text sequences.
            only_fc: Whether to use only the classification head.
            only_feat: Whether to return only the pooled features.
            return_embed: Whether to return the word embeddings.

        Returns:
            A dictionary containing the logits and/or pooled features, depending on the
            flags passed.
        """
        if only_fc:
            logits = self.classifier(x)
            return logits
        
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
        drop_hidden = self.dropout(last_hidden)
        pooled_output = torch.mean(drop_hidden, 1)
        
        if only_feat:
            return pooled_output
        
        logits = self.classifier(pooled_output)
        result_dict = {'logits':logits, 'feat':pooled_output}

        if return_embed:
            result_dict['embed'] = out_dict['hidden_states'][0]
            
        return result_dict
        
        
    def extract(self, x):
        """
        Extract the pooled features from a batch of text sequences.

        Args:
            x: Input tensor, a batch of text sequences.

        Returns:
            A tensor containing the pooled features for each text sequence.
        """
        out_dict = self.bert(**x, output_hidden_states=True, return_dict=True)
        last_hidden = out_dict['last_hidden_state']
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
        matcher = dict(stem=r'^{}bert.embeddings'.format(prefix), blocks=r'^{}bert.encoder.layer.(\d+)'.format(prefix))
        return matcher

    def no_weight_decay(self):
        return []



def bert_base_cased(pretrained=True, pretrained_path=None, **kwargs):
    """
    Loads a pre-trained BERT base cased model for classification.

    Args:
        pretrained: Whether to load a pre-trained model. Defaults to True.
        pretrained_path: The path to the pre-trained model. If not provided, the
        model will be loaded from the Hugging Face Transformers hub.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        A BERT base cased model.
    """
    model = ClassificationBert(name='bert-base-cased', **kwargs)
    return model


def bert_base_uncased(pretrained=True, pretrained_path=None, **kwargs):
    """
    Loads a pre-trained BERT base uncased model for classification.

    Args:
        pretrained: Whether to load a pre-trained model. Defaults to True.
        pretrained_path: The path to the pre-trained model. If not provided, the
        model will be loaded from the Hugging Face Transformers hub.
        **kwargs: Additional keyword arguments to pass to the model.

    Returns:
        A BERT base uncased model.
    """
    model = ClassificationBert(name='bert-base-uncased', **kwargs)
    return model
