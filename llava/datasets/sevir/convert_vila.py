import json
import os
from typing import Union, Dict, Sequence, Tuple, List
import math
from matplotlib import pyplot as plt
from matplotlib import colors
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import numpy as np
import datetime
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset as TorchDataset, DataLoader, random_split
from torchvision import transforms
from einops import rearrange
from .sevir_dataloader import (
    SEVIRDataLoader,
    SEVIR_RAW_SEQ_LEN,
    SEVIR_CATALOG,
    SEVIR_DATA_DIR,
)
from ..path import default_dataset_sevir_dir, default_dataset_dir


def ordinal(n: int):
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    else:
        suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    return str(n) + suffix


def seq_to_answer(seq, in_len, seq_len):
    r"""

    Parameters
    ----------
    seq: np.ndarray
        shape = (h w t)

    Returns
    -------
    answer: float
    """
    if in_len > seq.shape[2]:
        raise ValueError(f"in_len={in_len} is larger than seq.shape[2]={seq.shape[2]}")
    if seq_len < in_len:
        raise ValueError(f"seq_len={seq_len} is smaller than in_len={in_len}")
    return torch.mean(seq[:, :, seq_len-1]).item()


def seq_to_last_frame(seq, in_len):
    r"""

    Parameters
    ----------
    seq: np.ndarray
        shape = (h w t)

    Returns
    -------
    last_frame: np.ndarray
        shape = (h w)
    """
    return torch.mean(seq[:, :, in_len - 1]).item()


def seq_to_avg_input(seq, in_len):
    r"""

    Parameters
    ----------
    seq: np.ndarray
        shape = (h w t)

    Returns
    -------
    avg_input: float
    """
    return torch.mean(seq[:, :, :in_len]).item()


def convert_sevir_vila(
    save_dir: str,
    json_file_name: str = "sevir_convert.json",
    question_file_name: str = "sevir_questions.jsonl",
    gt_file_name: str = "sevir_gt.json",
    sevir_cfg: Dict = None,
    json_indent=4,
    **kwargs,
):
    seq_len = sevir_cfg["seq_len"]
    in_len = sevir_cfg["in_len"]
    start_date = datetime.datetime(*sevir_cfg["start_date"]) \
        if sevir_cfg["start_date"] is not None else None
    end_date = datetime.datetime(*sevir_cfg["end_date"]) \
        if sevir_cfg["end_date"] is not None else None

    # description = (f"There is a length-{in_len} sequence of rainfall frames from SEVIR dataset. "
    #                f"The pixel values in each frame range from 0 to 1, representing the intensity of rainfall, with higher values indicating greater rainfall intensity. ")
    description = (f"Given a sequence of {in_len} vertically integrated liquid (VIL) intensity frames from the SEVIR dataset, "
                   f"please analyze the progression of VIL intensity over time. "
                   f"Each frame is represented by pixel values ranging from 0 to 1, where 0 indicates no VIL and 1 represents maximum VIL intensity. "
                   )
    for i in range(in_len):
        description += f"<image> is the {ordinal(i + 1)} frame. "
    description += f"These frames are sequential with equal time intervals. "

    # question = (f"Given the sequence of frames, predict the average intensity of the future length-{seq_len-in_len} frames. "
    #             f"The response should be a single float number.")
    question = (f"Based on the provided sequence of frames, calculate and predict the average VIL intensity of the {ordinal(seq_len)} frame. "
                f"The response should be a single float number.")

    frame_stride = sevir_cfg["frame_stride"]
    sevir_dataloader = SEVIRDataLoader(
        data_types=["vil", ],
        seq_len=seq_len*frame_stride,
        raw_seq_len=SEVIR_RAW_SEQ_LEN,
        sample_mode="sequent",
        stride=sevir_cfg["stride"]*frame_stride,
        batch_size=1,
        layout="NHWT",
        num_shard=1,
        rank=0,
        split_mode="uneven",
        sevir_catalog=SEVIR_CATALOG,
        sevir_data_dir=SEVIR_DATA_DIR,
        start_date=start_date,
        end_date=end_date,
        datetime_filter=None,
        catalog_filter="default",
        shuffle=sevir_cfg["shuffle"],
        shuffle_seed=sevir_cfg["shuffle_seed"],
        output_type=np.float32,
        preprocess=True,
        rescale_method="01",
        downsample_dict=None,
        verbose=True,
    )
    abs_save_dir = os.path.join(default_dataset_dir, save_dir)
    if os.path.exists(abs_save_dir):
        raise FileExistsError(f"{abs_save_dir} already exists")
    os.makedirs(abs_save_dir)
    # sevir_dataloader.reset(shuffle=sevir_cfg["shuffle"])
    json_data = []
    question_data = []
    gt_data = []  # save the true answers for evaluation
    for index, data_dict in enumerate(tqdm(sevir_dataloader)):
        seq = data_dict["vil"].squeeze(0)
        seq = seq[..., ::frame_stride]

        answer = seq_to_answer(seq=seq, in_len=in_len, seq_len=seq_len)
        last_frame = seq_to_last_frame(seq=seq, in_len=in_len)
        avg_input = seq_to_avg_input(seq=seq, in_len=in_len)
        json_data.append(
            {
                "id": f"{index}",
                "image": index,
                "text": f"{description}{question}{answer:.4f}",
            }
        )
        question_data.append(
            {
                "question_id": f"{index}",
                "image": index,
                "text": f"{description}{question}",
                "category": "default",
            }
        )
        gt_data.append(
            {
                "question_id": f"{index}",
                "image": index,
                "answer": answer,
                "last_frame": last_frame,
                "avg_input": avg_input,
            }
        )
    with open(os.path.join(abs_save_dir, json_file_name), "w") as f:
        json.dump(json_data, f, indent=json_indent)
        f.close()
    with open(os.path.join(abs_save_dir, question_file_name), 'w') as f:
        for entry in question_data:
            json.dump(entry, f)
            f.write('\n')
        f.close()
    with open(os.path.join(abs_save_dir, gt_file_name), "w") as f:
        json.dump(gt_data, f, indent=json_indent)
        f.close()
