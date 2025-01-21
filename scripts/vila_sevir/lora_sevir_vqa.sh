#!/bin/bash

DATASET_NAME="tmp_sevir"

python ./scripts/vila_sevir/sevir_vqa.py \
    --model-path ./checkpoints/tmp_vila \
    --model-base Efficient-Large-Model/VILA-7B \
    --conv-mode vicuna_v1 \
    --sevir-cfg ./playground/data/$DATASET_NAME/cfg.yaml \
    --question-file ./playground/data/$DATASET_NAME/sevir_questions.jsonl \
    --answers-file ./playground/data/$DATASET_NAME/tmp_vila_answer.jsonl
