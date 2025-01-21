import json
from omegaconf import OmegaConf
import argparse
import os
from shutil import copyfile
from typing import Union, Dict, Sequence, Tuple, List
import math
from matplotlib import pyplot as plt
import numpy as np
import datetime
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange

from llava.datasets.path import root_dir, default_dataset_sevir_dir, default_dataset_dir
from llava.datasets.sevir.convert_llava_last_frame import convert_sevir_llava_last_frame


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_sevir_convert_last_frame', type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    save_dir = args.save
    sevir_cfg_path = os.path.join(root_dir, "scripts", "sevir", "sevir_cfg.yaml")

    json_file_name = "sevir_llava.json"
    question_file_name = "sevir_questions.jsonl"
    gt_file_name = "sevir_gt.json"

    convert_sevir_llava_last_frame(
        save_dir=save_dir,
        json_file_name=json_file_name,
        question_file_name=question_file_name,
        gt_file_name=gt_file_name,
        img_format="jpg",
        sevir_cfg=OmegaConf.to_object(OmegaConf.load(open(sevir_cfg_path, "r"))),
        json_indent=4,
    )
    cfg_target_path = os.path.join(default_dataset_dir, save_dir, "cfg.yaml")
    if ((not os.path.exists(cfg_target_path)) or
        (not os.path.samefile(sevir_cfg_path, cfg_target_path))):
        copyfile(sevir_cfg_path, cfg_target_path)
