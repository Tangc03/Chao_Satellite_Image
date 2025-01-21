import json
import re
from omegaconf import OmegaConf
import os
import argparse
from tqdm import tqdm
from typing import Union, Dict, Sequence, Tuple, List
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import pandas as pd
import torch

from llava.datasets.sevir.sevir_dataloader import (
    SEVIRDataLoader,
    SEVIR_RAW_SEQ_LEN,
    SEVIR_CATALOG,
    SEVIR_DATA_DIR,
)
from llava.datasets.path import default_dataset_sevir_dir, default_dataset_dir


def isfloat(val):
    try:
        float(val)
        return True
    except ValueError:
        return False


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tmp_sevir_convert', type=str)
    parser.add_argument('--answer', default=None, type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    data_dir = args.data
    cfg_path = os.path.join(default_dataset_dir, data_dir, "cfg.yaml")
    sevir_cfg = OmegaConf.load(open(cfg_path, "r"))

    if args.answer is not None:
        answer_file = os.path.join(default_dataset_dir, data_dir, args.answer)
        with open(answer_file, "r") as f:
            answer = [json.loads(line) for line in f]

    json_file_name = "sevir_llava.json"
    question_file_name = "sevir_questions.jsonl"
    data_file = os.path.join(default_dataset_dir, data_dir, json_file_name)
    question_file = os.path.join(default_dataset_dir, data_dir, question_file_name)
    with open(data_file, "r") as f:
        data = json.load(f)
    with open(question_file, "r") as f:
        question = [json.loads(line) for line in f]

    answer_mse = 0.0
    answer_mae = 0.0
    answer_no_float = 0
    answer_not_a_single_float = 0

    for i, ele in enumerate(tqdm(data)):
        target = float(ele["conversations"][1]["value"])

        if args.answer is not None:
            answer_ele = answer[i].get("text", "")
            if isfloat(answer_ele):
                answer_num = float(answer_ele)
            else:
                answer_num = re.findall(r"[-+]?(?:\d*\.*\d+)", answer_ele)
                if answer_num == []:
                    answer_no_float += 1
                    answer_num = 0.0
                else:
                    answer_not_a_single_float += 1
                    answer_num = 0.0
            answer_mse += (float(answer_num) - target) ** 2
            answer_mae += abs(float(answer_num) - target)

    if args.answer is not None:
        answer_mse /= len(data)
        answer_mae /= len(data)
        print(f"answer_mse = {answer_mse}")
        print(f"answer_mae = {answer_mae}")
        print(f"answer_no_float = {answer_no_float}")
        print(f"answer_not_a_single_float = {answer_not_a_single_float}")
