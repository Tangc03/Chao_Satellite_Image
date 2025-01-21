import json
import argparse
import os
from shutil import copyfile
from typing import Union, Dict, Sequence, Tuple, List

from llava.datasets.path import root_dir, default_exps_dir


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_split_prefix', type=str)
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    save_dir = args.save
    save_dir = os.path.join(default_exps_dir, save_dir)
    # list all dirs starting with "checkpoints-"
    all_checkpoints = os.listdir(save_dir)
    all_checkpoints = [ele for ele in all_checkpoints if ele.startswith("checkpoint-")]
    # generate symlink to "config.json" and "non_lora_trainables.bin" in save_dir to all checkpoints dirs
    for checkpoint in all_checkpoints:
        checkpoint_dir = os.path.join(save_dir, checkpoint)
        os.symlink(os.path.join(save_dir, "config.json"), os.path.join(checkpoint_dir, "config.json"))
        os.symlink(os.path.join(save_dir, "non_lora_trainables.bin"), os.path.join(checkpoint_dir, "non_lora_trainables.bin"))
