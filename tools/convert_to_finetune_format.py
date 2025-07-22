# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import tqdm
import argparse

def format_dataset(input_path, max_length=512):
    """
    Format a dataset by selecting a specific maximum length of text.

    Args:
        input_path (str): The path to the directory containing the original dataset.
        max_length (int, optional): The maximum length of text to keep (default: 512).

    Example:
        ```python
        format_dataset('data', max_length=512)
        ```
    """
    train_file = os.path.join(input_path, 'train.json')
    dev_file = os.path.join(input_path, 'dev.json')
    test_file = os.path.join(input_path, 'test.json')

    # if not os.path.exists(os.path.join(input_path)):
    #     os.mkdir(os.path.join(input_path))
    output_train_file = os.path.join(input_path, 'train_ft.json')
    output_dev_file = os.path.join(input_path, 'dev_ft.json')
    output_test_file = os.path.join(input_path, 'test_ft.json')

    for input_file, output_file in zip([train_file, dev_file, test_file], [output_train_file, output_dev_file, output_test_file]):
        with open(input_file, 'r') as f:
            data = json.load(f)
        print('Processing {} into {}.'.format(input_file, output_file))
        with open(output_file, 'w') as f:
            for doc_id, doc_item in tqdm.tqdm(data.items()):
                sample = {'text': doc_item['ori'], 'label': doc_item['label']}
                f.write(json.dumps(sample) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", default="data", type=str, help="The path of train, dev, and test files."
    )
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="One of task names: aclImdb ag_news amazon_review yahoo_answers yelp_review."
    )
    args = parser.parse_args()
    format_dataset(os.path.join(args.file_path, args.task_name))
