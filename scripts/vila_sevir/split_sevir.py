import json
from omegaconf import OmegaConf
import argparse
import os
from shutil import copyfile
from typing import Union, Dict, Sequence, Tuple, List
import math
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import datetime
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange

from llava.datasets.path import root_dir, default_dataset_sevir_dir, default_dataset_dir
from llava.datasets.sevir.convert_llava_multi_out import convert_sevir_llava_multi_out


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='tmp_sevir_convert', type=str)
    parser.add_argument('--save', default='tmp_split_prefix', type=str)
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--json_indent', default=4, type=int)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    save_dir = args.save
    data_dir = args.data
    save_train_dir = os.path.join(default_dataset_dir, f"{save_dir}_train")
    save_test_dir = os.path.join(default_dataset_dir, f"{save_dir}_test")
    if os.path.exists(save_train_dir) or os.path.exists(save_test_dir):
        raise ValueError(f"save dirs {save_train_dir} and {save_train_dir} already exist!")
    os.makedirs(save_train_dir)
    os.makedirs(save_test_dir)

    json_file_name = "sevir_vila.json"
    question_file_name = "sevir_questions.jsonl"
    gt_file_name = "sevir_gt.json"

    cfg_path = os.path.join(default_dataset_dir, data_dir, "cfg.yaml")
    data_file = os.path.join(default_dataset_dir, data_dir, json_file_name)
    question_file = os.path.join(default_dataset_dir, data_dir, question_file_name)
    gt_file = os.path.join(default_dataset_dir, data_dir, gt_file_name)

    target_cfg_path = os.path.join(default_dataset_dir, save_train_dir, "cfg.yaml")
    copyfile(cfg_path, target_cfg_path)
    target_cfg_path = os.path.join(default_dataset_dir, save_test_dir, "cfg.yaml")
    copyfile(cfg_path, target_cfg_path)

    print("Loading data...")
    with open(data_file, "r") as f:
        data = json.load(f)
    with open(question_file, "r") as f:
        question = [json.loads(line) for line in f]
    with open(gt_file, "r") as f:
        gt = json.load(f)
    print("Data loaded.")

    dataset_size = len(question)
    train_size = math.ceil(dataset_size * args.ratio)
    test_size = dataset_size - train_size
    train_indices = np.sort(np.random.choice(dataset_size, train_size, replace=False))
    test_indices = np.setdiff1d(np.arange(dataset_size), train_indices)

    train_data = [data[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    train_question = [question[i] for i in train_indices]
    test_question = [question[i] for i in test_indices]
    train_gt = [gt[i] for i in train_indices]
    test_gt = [gt[i] for i in test_indices]

    json_indent = args.json_indent
    with open(os.path.join(default_dataset_dir, save_train_dir, json_file_name), "w") as f:
        json.dump(train_data, f, indent=json_indent)
    print("Train data saved.")
    with open(os.path.join(default_dataset_dir, save_test_dir, json_file_name), "w") as f:
        json.dump(test_data, f, indent=json_indent)
    print("Test data saved.")
    with open(os.path.join(default_dataset_dir, save_train_dir, question_file_name), "w") as f:
        for line in train_question:
            json.dump(line, f)
            f.write('\n')
    print("Train question saved.")
    with open(os.path.join(default_dataset_dir, save_test_dir, question_file_name), "w") as f:
        for line in test_question:
            json.dump(line, f)
            f.write('\n')
    print("Test question saved.")
    with open(os.path.join(default_dataset_dir, save_train_dir, gt_file_name), "w") as f:
        json.dump(train_gt, f, indent=json_indent)
    print("Train gt saved.")
    with open(os.path.join(default_dataset_dir, save_test_dir, gt_file_name), "w") as f:
        json.dump(test_gt, f, indent=json_indent)
    print("Test gt saved.")

    # # generate symbolic links for images folder
    # images_data_dir = os.path.join(default_dataset_dir, data_dir, "images")
    # images_train_dir = os.path.join(default_dataset_dir, save_train_dir, "images")
    # images_test_dir = os.path.join(default_dataset_dir, save_test_dir, "images")
    # os.symlink(images_data_dir, images_train_dir)
    # os.symlink(images_data_dir, images_test_dir)
