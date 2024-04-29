# Original Copyright (c) Microsoft Corporation. Licensed under the MIT License.
# Modifications Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import argparse
import json
import torch
from tqdm import tqdm

MODE = 'dev'
# Load translation model

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')

en2ru = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
ru2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')


en2de.cuda()
de2en.cuda()

en2ru.cuda()
ru2en.cuda()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def truncate_sentence(s, max_length=1024):
    # Only keep the last max_length words
    return s[-max_length: ]


def cut_sentence(s):
    # remove the first 100 words
    s = s[::-1].rsplit(' ',100)[0][::-1]
    return s


def back_translation_augmentation(train_file):
    """
    Perform data augmentation through back-translation.

    Args:
        train_file (str): The path to the JSON file containing the training data.

    Returns:
        dict: The augmented training data.

    Example:
        ```
        augmented_data = back_translation_augmentation('train.json')
        ```
    """
    with open(train_file, 'r') as f:
        train_data = json.load(f)

    batchsize = 1024

    data = {}
    idx = 0    
    
    ori_sen = list(map(lambda x: x['ori'], train_data.values()))
    label = list(map(lambda x: x['label'], train_data.values()))
    
    ori_sen_list = list(chunks(ori_sen, batchsize))
    label_list = list(chunks(label, batchsize))

    for i in tqdm(range(len(ori_sen_list))):
        cur_ori_sen = ori_sen_list[i]
        cur_label = label_list[i]
        flag = True

        while flag:
            try:
                cur_ori_sen = list(map(truncate_sentence, cur_ori_sen))
                cur_aug_sen_0 = de2en.translate(en2de.translate(cur_ori_sen,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
                cur_aug_sen_1 = ru2en.translate(en2ru.translate(cur_ori_sen,  sampling=True, temperature=0.9),  sampling=True, temperature=0.9)
                flag = False
            except:
                longest_idx = cur_ori_sen.index(max(cur_ori_sen, key=len))
                shorter_sentence = cut_sentence(cur_ori_sen[longest_idx])
                cur_ori_sen[longest_idx] = shorter_sentence

        for j in range(len(cur_ori_sen)):
            data[str(idx)]={}
            data[str(idx)]['ori'] = cur_ori_sen[j]
            data[str(idx)]['aug_0'] = cur_aug_sen_0[j]
            data[str(idx)]['aug_1'] = cur_aug_sen_1[j]
            data[str(idx)]['label'] = cur_label[j]
            idx = idx + 1
            
    with open(train_file, 'w') as f_out:
        json.dump(data, f_out, indent=4)
        
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", default="data/SST-2", type=str, help="The path of train, dev, and test files."
    )
    args = parser.parse_args()
    
    back_translation_augmentation(os.path.join(args.file_path, 'train.json'))
