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


def seq_to_img(
    seq,
    in_len,
    use_grid=False,
    grid_linewidth=5,
    grid_color=(1, 0, 0),
    use_number=False,
    number_size=30,
    number_color=(1, 0, 0),
):
    r"""

    Parameters
    ----------
    seq: np.ndarray
        shape = (h w t)

    Returns
    -------
    img: np.ndarray
        shape = (h w)
    """
    if in_len > seq.shape[2]:
        raise ValueError(f"in_len={in_len} is larger than seq.shape[2]={seq.shape[2]}")
    save_img_size_mult = math.ceil(math.sqrt(in_len))
    img_h = seq.shape[0] * save_img_size_mult
    img_w = seq.shape[1] * save_img_size_mult
    # frames_per_img = save_img_size_mult ** 2
    img = np.zeros((img_h, img_w, 3))
    if use_number:
        if isinstance(number_color, str):
            number_color = colors.to_rgb(number_color)
        number_color = tuple([int(255 * c) for c in number_color])
    for i in range(in_len):
        img_i = i // save_img_size_mult
        img_j = i % save_img_size_mult
        img[img_i * seq.shape[0]: (img_i + 1) * seq.shape[0],
        img_j * seq.shape[1]: (img_j + 1) * seq.shape[1], :] = np.stack([seq[:, :, i], ] * 3, axis=2)
    if use_grid:
        if isinstance(grid_color, str):
            grid_color = colors.to_rgb(grid_color)
        grid_linewidth //= 2
        for i in range(1, save_img_size_mult):
            img[max(i * seq.shape[0] - grid_linewidth, 0): min(i * seq.shape[0] + grid_linewidth, img_h), :, :] = grid_color
            img[:, max(i * seq.shape[1] - grid_linewidth, 0): min(i * seq.shape[1] + grid_linewidth, img_w), :] = grid_color
    if use_number:
        img_number = np.zeros((img_h, img_w, 3))
        img_number = Image.fromarray((img_number * 255).astype(np.uint8))
        draw = ImageDraw.Draw(img_number)
        font = ImageFont.load_default(number_size)
        for i in range(in_len):
            img_i = i // save_img_size_mult
            img_j = i % save_img_size_mult
            draw.text((img_j * seq.shape[1] + 5, img_i * seq.shape[0]),
                      str(i + 1),
                      font=font, fill=number_color)
        img_number = np.array(img_number).astype(float) / 255.0
        img_pos = img_number.astype(bool)
        for i in range(3):
            img_pos[:, :, i] = np.sum(img_pos, axis=2)
        img = img * (1 - img_pos) + img_number
    return img


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
    out_len = seq_len - in_len
    answer = ""
    for i in range(out_len - 1):
        answer += f"{torch.mean(seq[:, :, in_len + i]).item():.4f},"
    answer += f"{torch.mean(seq[:, :, seq_len - 1]).item():.4f}"
    return answer


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


def convert_sevir_llava_multi_out(
    save_dir: str,
    json_file_name: str = "sevir_convert.json",
    question_file_name: str = "sevir_questions.jsonl",
    gt_file_name: str = "sevir_gt.json",
    img_format="jpg",
    sevir_cfg: Dict = None,
    json_indent=4,
    use_grid=False,
    grid_linewidth=5,
    grid_color=(1, 0, 0),
    use_number=False,
    number_size=5,
    number_color=(1, 0, 0),
):
    seq_len = sevir_cfg["seq_len"]
    in_len = sevir_cfg["in_len"]
    out_len = sevir_cfg["out_len"]
    assert seq_len == in_len + out_len

    start_date = datetime.datetime(*sevir_cfg["start_date"]) \
        if sevir_cfg["start_date"] is not None else None
    end_date = datetime.datetime(*sevir_cfg["end_date"]) \
        if sevir_cfg["end_date"] is not None else None

    save_img_size_mult = math.ceil(math.sqrt(in_len))
    # description = (f"The input image contains a length-{in_len} sequence of rainfall frames from SEVIR dataset. "
    #                f"These {in_len} frames are arranged in a {save_img_size_mult}x{save_img_size_mult} grid following the temporal order, from left to right, top to bottom, ")
    #                f"The pixel values in each frame range from 0 to 1, representing the intensity of rainfall, with higher values indicating greater rainfall intensity. ")
    description = (f"The input image contains a sequence of {in_len} vertically integrated liquid (VIL) intensity frames from the SEVIR dataset. "
                   f"Please analyze the progression of VIL intensity over time. "
                   f"Each frame is represented by pixel values ranging from 0 to 1, where 0 indicates no VIL and 1 represents maximum VIL intensity. "
                   f"These {in_len} frames are arranged in a {save_img_size_mult}x{save_img_size_mult} grid following the temporal order, from left to right, top to bottom, ")
    if use_grid:
        if isinstance(grid_color, str):
            description += f"with a {grid_color} grid separating each frame. "
        else:
            raise NotImplementedError("grid_color must be a string")
    if use_number:
        if isinstance(number_color, str):
            description += f"and each frame is labeled with a {number_color} number in the upper left corner indicating its temporal order. "
        else:
            raise NotImplementedError("number_color must be a string")
    # question = (f"Given the image containing a length-{in_len} sequence of frames, predict the average intensity of the future length-{seq_len-in_len} frames. "
    #             f"The answer should be a single float number.")
    #             # f"The answer should be a single float number in the range of 0.0 to 1.0 . ")
    question = f"Based on the provided sequence of frames, calculate and predict the average VIL intensity of "
    if out_len == 1:
        question += f"the next frame. The response should be a single float number."
        # question += f"the {ordinal(seq_len)}"
    else:
        for i in range(out_len - 1):
            question += f"the {ordinal(in_len + i + 1)}, "
        question += f"and the {ordinal(seq_len)}"
        question += f" frame. The response should be a sequence of float number separated by commas."

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
    image_abs_save_dir = os.path.join(abs_save_dir, "images")
    image_save_dir = os.path.join(save_dir, "images")
    if os.path.exists(image_abs_save_dir):
        raise ValueError(f"image_save_dir {image_abs_save_dir} already exists")
    os.makedirs(image_abs_save_dir)
    # sevir_dataloader.reset(shuffle=sevir_cfg["shuffle"])
    json_data = []
    question_data = []
    gt_data = []  # save the true answers for evaluation
    for index, data_dict in enumerate(tqdm(sevir_dataloader)):
        seq = data_dict["vil"].squeeze(0)
        seq = seq[..., ::frame_stride]
        img = seq_to_img(
            seq=seq, in_len=in_len,
            use_grid=use_grid,
            grid_linewidth=grid_linewidth,
            grid_color=grid_color,
            use_number=use_number,
            number_size=number_size,
            number_color=number_color,
            )
        answer = seq_to_answer(seq=seq, in_len=in_len, seq_len=seq_len)
        last_frame = seq_to_last_frame(seq=seq, in_len=in_len)
        avg_input = seq_to_avg_input(seq=seq, in_len=in_len)
        img_abs_path = os.path.join(image_abs_save_dir, f"{index}.{img_format}")
        img_path = os.path.join(image_save_dir, f"{index}.{img_format}")
        # plt.imsave(img_abs_path, img, cmap="gray")
        plt.imsave(img_abs_path, img)
        json_data.append(
            {
                "id": f"{index}",
                "image": f"{img_path}",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"{description}<image>{question}"
                    },
                    {
                        "from": "gpt",
                        "value": f"{answer}"
                    },
                ]
            }
        )
        question_data.append(
            {
                "question_id": f"{index}",
                "image": f"{img_path}",
                "text": f"{description}{question}",
                "category": "default",
            }
        )
        gt_data.append(
            {
                "question_id": f"{index}",
                "image": f"{img_path}",
                "answer": f"{answer}",
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
