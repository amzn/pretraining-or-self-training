# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import tqdm
import shutil
import argparse
import collections

def format_dataset(file_path, task_name, train_size, valid_size, test_size, keep_original_dev_test=False):
    num_classes = {
        'aclImdb': 2,
        'SST-2': 2,
        'ag_news': 4,
        'amazon_review': 5,
        'yelp_review': 5,
        'yahoo_answers': 10
    }
    num_class = num_classes[task_name]
    train_num_each_class = train_size / num_class
    assert train_num_each_class == int(train_num_each_class)
    if not keep_original_dev_test:
        valid_num_each_class = valid_size / num_class
        test_num_each_class = test_size / num_class
        assert valid_num_each_class == int(valid_num_each_class)
        assert test_num_each_class == int(test_num_each_class)
    else:
        valid_num_each_class = test_num_each_class = 0

    input_path = os.path.join(file_path, task_name)
    if not keep_original_dev_test:
        output_path = os.path.join(file_path, task_name + '_' + str(train_size) + '_' + str(valid_size) + '_' + str(test_size))
    else:
        output_path = os.path.join(file_path, task_name + '_' + str(train_size))
    if not os.path.exists(os.path.join(output_path)):
        os.mkdir(os.path.join(output_path))
        
    train_file = os.path.join(input_path, 'train.json')
    dev_file = os.path.join(input_path, 'dev.json')
    test_file = os.path.join(input_path, 'test.json')

    output_train_file = os.path.join(output_path, 'train.json')
    output_dev_file = os.path.join(output_path, 'dev.json')
    output_test_file = os.path.join(output_path, 'test.json')

    for input_file, output_file, num_each_class in zip([train_file, dev_file, test_file], [output_train_file, output_dev_file, output_test_file], [train_num_each_class, valid_num_each_class, test_num_each_class]):
        if keep_original_dev_test and ('dev' in input_file or 'test' in input_file):
            shutil.copyfile(input_file, output_file)
        else:
            with open(input_file, 'r') as f:
                data = json.load(f)

            output = [{} for _ in range(num_class)]
            current_id = 0
            for doc_id, doc_item in tqdm.tqdm(data.items()):
                if len(output[int(doc_item['label'])]) < num_each_class:
                    output[int(doc_item['label'])][current_id] = doc_item
                    current_id += 1
            
            print('Number of samples for each class:', [len(d) for d in output])
            output_dct = {k: v for d in output for k, v in d.items()}
            print('Processing {} into {} with {} samples.'.format(input_file, output_file, len(output_dct)))
            with open(output_file, 'w') as f:
                json.dump(collections.OrderedDict(sorted(output_dct.items())), f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path", default="data", type=str, help="The path of train, dev, and test files."
    )
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="One of task names: aclImdb ag_news amazon_review yahoo_answers yelp_review."
    )
    parser.add_argument(
        "--train_size", default=None, type=int, required=True, help="The size of the train set."
    )
    parser.add_argument(
        "--valid_size", default=None, type=int, help="The size of the valid set."
    )
    parser.add_argument(
        "--test_size", default=None, type=int, help="The size of the test set."
    )
    parser.add_argument(
        "--keep_original_dev_test", action='store_true', help="Whether to keep the original dev and test set."
    )
    args = parser.parse_args()
    format_dataset(args.file_path, args.task_name, args.train_size, args.valid_size, args.test_size, args.keep_original_dev_test)
