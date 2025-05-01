# Rethinking Semi-supervised Learning with Language Models
This repository contains the code for the paper titled **[Rethinking Semi-supervised Learning with Language Models]()**, built upon [huggingface](https://github.com/huggingface/transformers) and [semilearn](https://github.com/microsoft/Semi-supervised-learning).


## Quick Links
- [Rethinking Semi-supervised Learning with Language Models](#rethinking-semi-supervised-learning-with-language-models)
  - [Quick Links](#quick-links)
  - [Overview](#overview)
  - [1. Installation](#1-installation)
  - [2. Preprocess the data](#2-preprocess-the-data)
  - [3. Task Adaptive Pre-Training](#3-task-adaptive-pre-training)
  - [4. Fine-tuning from the roberta-base or the pre-trained checkpoints](#4-fine-tuning-from-the-roberta-base-or-the-pre-trained-checkpoints)
  - [5. Self Training](#5-self-training)
  - [Bugs or questions?](#bugs-or-questions)
  - [Citation](#citation)
  - [Authors](#authors)
  - [Security](#security)
  - [License](#license)

## Overview
You can reproduce the continued pre-training and self training experiments of our recent paper [Rethinking Semi-supervised Learning with Language Models]().

## 1. Installation
```sh
conda create --name storpt python=3.8
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 2. Preprocess the data
The dataset can be downloaded via the following repository: [semilearn](https://github.com/microsoft/Semi-supervised-learning/tree/main/preprocess). To preprocess data for language modeling, follow these steps:
- Set TASK_NAME as one of the following: aclImdb, ag_news, amazon_review, yahoo_answers, or amazon_review.
- Run the provided code snippets for the desired format.

For task adaptive pre-training format:
```bash
for TASK_NAME in aclImdb ag_news amazon_review yahoo_answers; do
    python tools/convert_to_pretrain_format.py --task_name ${TASK_NAME};
done
```
For the format of the text classification:
```bash
for TASK_NAME in aclImdb ag_news amazon_review yahoo_answers; do
    python tools/convert_to_finetune_format.py --task_name ${TASK_NAME};
done
```
For text classification with a partially labeled dataset (Use the index of labeled data to ensure fair comparison with semi-supervised approaches):
```bash
TASK_NAME=aclImdb
LABEL_SIZE=20
python tools/convert_to_finetune_semi_format.py --num_labels ${LABEL_SIZE} --task_name ${TASK_NAME}
```
To change the size of the dataset:
```bash
TASK_NAME=amazon_review
TRAIN_SIZE=23000
VAL_SIZE=2000
TEST_SIZE=25000
python tools/convert_dataset_size.py --task_name ${TASK_NAME} --train_size ${TRAIN_SIZE} --valid_size ${VAL_SIZE} --test_size ${TEST_SIZE}
python tools/convert_dataset_size.py --task_name ${TASK_NAME} --train_size ${TRAIN_SIZE} --keep_original_dev_test
```

## 3. Task Adaptive Pre-Training
The code for masked language modeling can be found in [run_mlm.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py). To perform task adaptive pre-training, execute the command below:
```bash
for TASK_NAME in aclImdb ag_news amazon_review yahoo_answers; do \
    python run_mlm.py \
        --model_name_or_path roberta-base \
        --train_file data/${TASK_NAME}/train.txt \
        --validation_file data/${TASK_NAME}/dev.txt \
        --line_by_line \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 1e-04 \
        --optim adamw_torch \
        --weight_decay 0.01 \
        --adam_beta1 0.9 \
        --adam_beta2 0.98 \
        --adam_epsilon 1e-06 \
        --do_train \
        --do_eval \
        --save_steps 500 \
        --evaluation_strategy steps \
        --eval_steps 500 \
        --num_train_epochs 100 \
        --warmup_ratio 0.06 \
        --mlm_probability 0.15 \
        --fp16 \
        --output_dir saved_tapt/${TASK_NAME} \
        --load_best_model_at_end; \
done
```
Note: The above command assumes training on 8x16GB V100 GPUs. Each GPU uses a batch size of 8 sequences and accumulates gradients for a total batch size of 256 sequences. If you have a GPU with mixed precision capabilities (architecture Pascal or more recent), you can use mixed precision training with PyTorch 1.6.0 or latest by adding the flag `--fp16` to the scripts mentioned above! The learning rate and batch size are closely related and should be adjusted together. More details can be found [here](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md).

## 4. Fine-tuning from the roberta-base or the pre-trained checkpoints
The code for masked language modeling can be found in [run_glue.py](https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py). To perform task adaptive fine-tuning, execute the command below:
```bash
TASK_NAME=
LABEL_SIZE=
CHECKPOINT_DIR=roberta-base
for seed in 1 2 3 4 5; do \
    CUDA_VISIBLE_DEVICES=0 python run_glue.py \
        --train_file data/${TASK_NAME}/labeled_idx/train_ft_label${LABEL_SIZE}_seed${seed}.json \
        --validation_file data/${TASK_NAME}/dev_ft.json \
        --test_file data/${TASK_NAME}/test_ft.json \
        --model_name_or_path ${CHECKPOINT_DIR} \
        --seed ${seed} \
        --do_train \
        --do_eval \
        --do_predict \
        --per_device_train_batch_size 16 \
        --per_device_eval_batch_size 16 \
        --max_seq_length 256 \
        --num_train_epochs 50 \
        --save_strategy epoch \
        --evaluation_strategy epoch \
        --learning_rate 2e-05 \
        --warmup_ratio 0.0 \
        --fp16 \
        --metric_for_best_model eval_f1 \
        --load_best_model_at_end \
        --save_total_limit 1 \
        --output_dir saved_finetuned/${TASK_NAME}_label${LABEL_SIZE}_seed${seed}; \
done
```

## 5. Self Training
Run self training
```bash
ALGORITHM=
TASK_NAME=
LABEL_SIZE=
for seed in 1 2 3 4 5; do \
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --seed ${seed} \
        --c config_roberta/${ALGORITHM}/${ALGORITHM}_${TASK_NAME}_${LABEL_SIZE}_0.yaml \
        --save_dir ./saved_models \
        --save_name ${ALGORITHM}_${TASK_NAME}_${LABEL_SIZE}_${seed} \
        --load_path ./saved_models/${ALGORITHM}_${TASK_NAME}_${LABEL_SIZE}_${seed}/latest_model.pth; \
done
```

Train models under the self-taught learning (STL) settings.
```bash
ALGORITHM=
TASK_NAME=
UNLABEL_DATASET=
LABEL_SIZE=
for seed in 1 2 3 4 5; do \
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --seed ${seed} \
        --num_labels ${LABEL_SIZE} \
        --dataset ${TASK_NAME} \
        --custom_unlabeled_data_file data/${UNLABEL_DATASET}/train.json \
        --c config_roberta/${ALGORITHM}/${ALGORITHM}_aclImdb_100_0.yaml \
        --save_dir saved_domainshift_stl \
        --save_name ${ALGORITHM}_${TASK_NAME}_${UNLABEL_DATASET}_${LABEL_SIZE}_${seed} \
        --load_path ./saved_domainshift_stl/${ALGORITHM}_${TASK_NAME}_${UNLABEL_DATASET}_${LABEL_SIZE}_${seed}/latest_model.pth; \
done
```

Train models under Unsupervised Domain Adaptation (UDA) settings.
```bash
ALGORITHM=
SOURCE_DATASET=
TARGET_DATASET=
LABEL_SIZE=
for seed in 1 2 3 4 5; do \
    CUDA_VISIBLE_DEVICES=0 python train.py \
        --seed ${seed} \
        --num_labels ${LABEL_SIZE} \
        --dataset ${SOURCE_DATASET} \
        --custom_unlabeled_data_file data/${TARGET_DATASET}/train.json \
        --custom_dev_data_file data/${TARGET_DATASET}/dev.json \
        --custom_test_data_file data/${TARGET_DATASET}/test.json \
        --c config_roberta/${ALGORITHM}/${ALGORITHM}_${SOURCE_DATASET}_${LABEL_SIZE}_0.yaml \
        --save_dir saved_domainshift_uda \
        --save_name ${ALGORITHM}_${SOURCE_DATASET}_${TARGET_DATASET}_${LABEL_SIZE}_${seed} \
        --load_path ./saved_domainshift_uda/${ALGORITHM}_${SOURCE_DATASET}_${TARGET_DATASET}_${LABEL_SIZE}_${seed}/latest_model.pth; \
done
```

## Bugs or questions?
If you have any inquiries pertaining to the code or the paper, please do not hesitate to contact [Zhengxiang Shi](https://zhengxiangshi.github.io/). In case you encounter any issues while utilising the code or wish to report a bug, you may open an issue. We kindly request that you provide specific details regarding the problem so that we can offer prompt and efficient assistance.

## Citation
```
@inproceedings{shi2023rethinking,
  title={Rethinking Semi-supervised Learning with Language Models},
  author={Shi, Zhengxiang and Tonolini, Francesco and Aletras, Nikolaos and Yilmaz, Emine and Kazai, Gabriella and Jiao, Yunlong},
  year={2023},
  address = {Toronto, Canada},
  booktitle = {Findings of the Association for Computational Linguistics: ACL 2023},
  publisher = {Association for Computational Linguistics},
}
```

## Authors

- [**Zhengxiang Shi**](https://profiles.ucl.ac.uk/83462-zhengxiang-shi): Main contributor

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
