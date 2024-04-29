# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import tqdm
import argparse
import numpy as np

def format_dataset(input_path, num_labels=20, max_length=512):
    """
    Format a dataset by selecting a specific number of labeled samples and a maximum text length.

    Args:
        input_path (str): The path to the directory containing the original dataset and labeled indices.
        num_labels (int, optional): The number of labeled samples to select (default: 20).
        max_length (int, optional): The maximum length of text to keep (default: 512).

    Example:
        format_dataset('data', num_labels=20, max_length=512)
    """
    train_file = os.path.join(input_path, 'train.json')

    with open(train_file, 'r') as f:
        data = json.load(f)
        
    for seed in [1, 2, 3, 4, 5]:
        lb_idx = np.load(os.path.join(input_path, "labeled_idx", "lb_labels{}_seed{}_idx.npy".format(num_labels, seed)))
        lb_idx_set = set(lb_idx.tolist())

        output_train_file = os.path.join(input_path, "labeled_idx", 'train_ft_label{}_seed{}.json'.format(num_labels, seed))
        print('Processing {} into {}.'.format(train_file, output_train_file))
        with open(output_train_file, 'w') as f:
            for idx, (doc_id, doc_item) in tqdm.tqdm(enumerate(data.items())):
                if idx in lb_idx_set:
                    sample = {'text': doc_item['ori'], 'label': doc_item['label']}
                    f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", default="data", type=str, help="The path of train, dev, and test files."
    )
    parser.add_argument(
        "--num_labels", default=20, type=int, help="The number of labels for semi supervised learning."
    )
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="One of task names: aclImdb ag_news amazon_review yahoo_answers yelp_review."
    )
    args = parser.parse_args()
    format_dataset(os.path.join(args.file_path, args.task_name), args.num_labels)
