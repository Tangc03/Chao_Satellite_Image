#!/bin/bash

python ./llava/eval/model_vqa.py \
    --model-path ./checkpoints/llava-v1.5-7b-sevir-lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --question-file ./playground/data/tmp_sevir_convert/sevir_questions.jsonl \
    --image-folder ./playground/data \
    --answers-file ./playground/data/tmp_sevir_convert/tmp_answer.jsonl
