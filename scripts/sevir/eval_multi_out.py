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
    gt_file_name = "sevir_gt.json"
    data_file = os.path.join(default_dataset_dir, data_dir, json_file_name)
    question_file = os.path.join(default_dataset_dir, data_dir, question_file_name)
    gt_file = os.path.join(default_dataset_dir, data_dir, gt_file_name)
    with open(data_file, "r") as f:
        data = json.load(f)
    with open(question_file, "r") as f:
        question = [json.loads(line) for line in f]
    with open(gt_file, "r") as f:
        gt = json.load(f)  # List[Dict[str, Union[str, float]]]

    last_frame_mse = 0.0
    avg_input_mse = 0.0
    last_frame_mae = 0.0
    avg_input_mae = 0.0
    answer_mse = 0.0
    answer_mae = 0.0
    answer_no_float = 0
    answer_not_a_single_float = 0

    for i, ele in enumerate(tqdm(gt)):
        target = ele["answer"]
        target = target.split(",")
        target = np.array([float(t) for t in target])
        last_frame = float(ele["last_frame"])
        avg_input = float(ele["avg_input"])
        last_frame_mse += np.sum((last_frame - target) ** 2) / len(target)
        avg_input_mse += np.sum((avg_input - target) ** 2) / len(target)
        last_frame_mae += np.sum(abs(last_frame - target)) / len(target)
        avg_input_mae += np.sum(abs(avg_input - target)) / len(target)

        if args.answer is not None:
            answer_ele = answer[i].get("text", "")
            answer_ele = answer_ele.split(",")
            answer_ele_num = [0.0, ] * 4
            all_are_numbers = True
            for i, answer_ele_single_num in enumerate(answer_ele):
                if isfloat(answer_ele_single_num):
                    answer_ele_num[i] = float(answer_ele_single_num)
                else:
                    all_are_numbers = False
                    answer_no_float += 1
                    # answer_ele_num.append(0.0)
            if not all_are_numbers:
                answer_not_a_single_float += 1
            # if isfloat(answer_ele):
            #     answer_num = float(answer_ele)
            # else:
            #     answer_num = re.findall(r"[-+]?(?:\d*\.*\d+)", answer_ele)
            #     if answer_num == []:
            #         answer_no_float += 1
            #         answer_num = 0.0
            #     else:
            #         answer_not_a_single_float += 1
            #         answer_num = 0.0
            answer_mse += np.sum((np.array(answer_ele_num) - target) ** 2) / len(target)
            answer_mae += np.sum(abs(np.array(answer_ele_num) - target)) / len(target)

    last_frame_mse /= len(gt)
    avg_input_mse /= len(gt)
    last_frame_mae /= len(gt)
    avg_input_mae /= len(gt)
    print(f"last_frame_mse = {last_frame_mse}")
    print(f"avg_input_mse = {avg_input_mse}")
    print(f"last_frame_mae = {last_frame_mae}")
    print(f"avg_input_mae = {avg_input_mae}")
    if args.answer is not None:
        answer_mse /= len(gt)
        answer_mae /= len(gt)
        print(f"answer_mse = {answer_mse}")
        print(f"answer_mae = {answer_mae}")
        print(f"answer_no_float = {answer_no_float}")
        print(f"answer_not_a_single_float = {answer_not_a_single_float}")
