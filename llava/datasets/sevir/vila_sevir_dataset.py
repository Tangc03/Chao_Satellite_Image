import os, os.path as osp
import base64
import copy
from dataclasses import dataclass, field
import io
import numpy as np
import random
import json
import logging
import pathlib
import pickle
import time
from typing import Dict, Optional, Sequence, List, Any
import re

import torch
#torch.backends.cudnn.enabled = False

import transformers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.train import TrainingArguments, DataArguments, preprocess_multimodal
from llava import conversation as conversation_lib
from llava.mm_utils import tokenizer_image_token, process_images

from PIL import Image

from omegaconf import OmegaConf
from ..vila.dataset import LazySupervisedDataset
from .vila_sevir_parser import SEVIRVILAParser


class LazySEVIRDataset(Dataset):
    r"""
    Adapted from LazyMMC4Dataset: https://github.com/Efficient-Large-Model/VILA/blob/ef662c84fe7e34101184ceab310fc41f837084b4/llava/data/dataset.py#L570
    """
    num_image_tokens = 576

    def __init__(
        self,
        data_path: str,
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments,
        image_following_text_only=False,
        text_only=False,
        # above from LazyMMC4Dataset
        sevir_cfg: Dict[str, Any] = None,
    ):
        super().__init__()
        if sevir_cfg is None:
            sevir_cfg_path = os.path.join(os.path.dirname(data_path), "cfg.yaml")
            sevir_cfg = OmegaConf.to_object(OmegaConf.load(sevir_cfg_path))
        self.sevir_parser = SEVIRVILAParser(
            cfg=sevir_cfg["sevir"],
            in_len=sevir_cfg["sevir"]["in_len"],
        )
        self.list_data_dict = json.load(open(data_path, "r"))

        # import pickle
        #
        # n_samples = []
        # # actually shards and stats info
        # n_shards = len(os.listdir(data_path)) // 2
        # # n_shards = 100
        # count_info_list = sorted(
        #     [f for f in os.listdir(data_path) if f.endswith(".count")]
        # )[:n_shards]
        # n_samples = [
        #     int(open(os.path.join(data_path, f), "r").read().strip())
        #     for f in count_info_list
        # ]
        #
        # print("total MMC4 samples", sum(n_samples))  # 10,881,869
        #
        # rank = training_args.process_index  # int(os.environ["RANK"])
        # world_size = training_args.world_size  # int(os.environ["WORLD_SIZE"])
        # shared_size = n_shards // world_size
        #
        # gpu_samples = [
        #     sum(n_samples[i * shared_size: (i + 1) * shared_size])
        #     for i in range(world_size)
        # ]
        # self.n_samples = min(gpu_samples) * world_size  # total size
        # self.idx_offset = rank * min(gpu_samples)
        # shard_start, shard_end = rank * shared_size, (rank + 1) * shared_size
        # print(f" * loading data from shard {shard_start}-{shard_end}")
        #
        # shard_names = [d.replace(".count", ".pkl") for d in count_info_list]
        # shard_names = shard_names[shard_start:shard_end]
        #
        # full_data_list = []
        # # now load data
        # for shard_name in shard_names:
        #     # load shard
        #     with open(os.path.join(data_path, shard_name), "rb") as f:
        #         data_list = pickle.load(f)
        #
        #     full_data_list.extend(data_list)
        #
        # print("* loaded totally {} samples".format(len(full_data_list)))

        # self.data_list = full_data_list

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.image_following_text_only = image_following_text_only
        self.text_only = text_only

    def __len__(self):
        # return len(self.data_list)
        # return self.n_samples
        return len(self.list_data_dict)

    # @property
    # def modality_lengths(self):
    #     # the original implementation from VILA
    #     # Estimate the number of tokens after tokenization, used for length-grouped sampling
    #     length_list = []
    #     for info in self.data_list:
    #         num_images = min(6, len(info["image_info"]))
    #         sentences = [info["text_list"][x["matched_text_index"]] for x in info["image_info"][:num_images]]
    #         # The unit of cur_len is "words". We assume 1 word = 2 tokens.
    #         cur_len = num_images * self.num_image_tokens // 2 + sum([len(x) for x in sentences])
    #         length_list.append(cur_len)
    #     return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            num_images = len([m.start() for m in re.finditer('<image>', sample['text'])])
            cur_len = num_images * self.num_image_tokens // 2 + len(sample['text'])
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO
        sources = self.list_data_dict[i]
        sevir_idx = sources["image"]
        images = self.sevir_parser.parse(idx=sevir_idx, ret_PIL=True)  # list of PIL images

        # sources = preprocess_multimodal(
        #     copy.deepcopy([e["conversations"] for e in sources]),
        #     self.data_args)  # LLaVA move "<image>" to the beginning of the sentence

        # sentences = info["text_list"]
        # # kentang-mit@: remove existing <image> tokens in the sentences
        # for ix in range(len(sentences)):
        #     # if this is an html tag, we still preserve its semantic meaning
        #     sentences[ix] = sentences[ix].replace("<image>", "<IMAGE>")
        # sim_matrix = info["similarity_matrix"]  # we do not use this...

        # convert images from base64 to PIL and filter based on image-text similarity
        # images, sentence_ixs = [], []
        # if not self.text_only:
        #     for sample_image, sim_vec in zip(info["image_info"], sim_matrix):
        #         image_base64 = sample_image["image_base64"]
        #         rawbytes = base64.b64decode(image_base64)
        #
        #         sim_ix = sample_image["matched_text_index"]
        #         # sim_ix = np.argmax(sim_vec)
        #         # sim_score = sim_vec[sim_ix]
        #
        #         # filter to images >= 5KB
        #         # if len(rawbytes) // 1000 <= 5:
        #         #     continue
        #         # if sim_score < 0.24:
        #         #     continue
        #         image = Image.open(io.BytesIO(rawbytes)).convert("RGB")
        #
        #         images.append(image)
        #         sentence_ixs.append(sim_ix)

        # constrain max num 6 images
        # max_num_images = 6
        # if len(images) > max_num_images:
        #     images = images[:max_num_images]
        #     sentence_ixs = sentence_ixs[:max_num_images]
        #
        # # reorder images according to text insertion
        # images = [images[iii] for iii in np.argsort(sentence_ixs)]
        #
        # # preprocess and tokenize text
        # for ix in sentence_ixs:
        #     sentences[ix] = f"<image>{sentences[ix]}"
        #
        # if self.image_following_text_only:
        #     # use pad tokens to divide sentence pieces
        #     text = self.tokenizer.pad_token.join(sentences)
        # else:
        #     text = " ".join(sentences)
        # # results in text == "<image>image_description <image>image_description <image>image_description"

        text = sources["text"]
        # whitespace cleanup
        text = text.replace("<image> ", "<image>").replace(" <image>", "<image>")
        text = f"{text}{self.tokenizer.eos_token}"  # add eos token

        if len(images) > 0:
            # images = torch.stack(
            #     [
            #         LazySupervisedDataset._process_image(
            #             image,
            #             self.data_args,
            #             self.image_folfer,
            #         )
            #         for image in images
            #     ]
            # )
            images = process_images(
                images=images,
                image_processor=self.data_args.image_processor,
                model_cfg=self.data_args,
            )  # (T, 3, H, W)

            # the same size for all images, so we concat
            # cur_token_len = (
            #     images[0].shape[-2] // self.multimodal_cfg["patch_size"]
            # ) * (images[0].shape[-1] // self.multimodal_cfg["patch_size"])
            # cur_token_len += self.multimodal_cfg["n_extra_patch"]
        else:
            images = None
            # cur_token_len = 0

        # im_patch_token = self.tokenizer.convert_tokens_to_ids(
        #     [DEFAULT_IMAGE_PATCH_TOKEN]
        # )[0]
        # print(text, len(images))
        input_ids = tokenizer_image_token(
            text,
            self.tokenizer,
            return_tensors="pt",
        )
        assert len(input_ids.shape) == 1

        # now check the case where the last token is image patch token
        if input_ids[-1] == IMAGE_TOKEN_INDEX:  # need to remove one last image
            last_non_im_patch_indices = torch.where(input_ids != IMAGE_TOKEN_INDEX)[0][-1] + 1
            input_ids = input_ids[:last_non_im_patch_indices]

        n_im_patch = (input_ids == IMAGE_TOKEN_INDEX).sum().item()

        images = images[: n_im_patch]
        assert len(images) == n_im_patch, print(text, input_ids)

        targets = input_ids.clone()

        if self.image_following_text_only:  # keep only text after leading image token
            # remove loss for any token before the first <image> token
            label_idx = 0
            while (
                label_idx < targets.shape[-1] and targets[label_idx] != IMAGE_TOKEN_INDEX
            ):
                targets[label_idx] = IGNORE_INDEX
                label_idx += 1

            pad_token = self.tokenizer.convert_tokens_to_ids(
                [self.tokenizer.pad_token]
            )[0]

            pad_token_idxs = torch.where(targets == pad_token)[0]
            for pad_token_idx in pad_token_idxs:
                token_idx = pad_token_idx + 1
                while (
                    token_idx < targets.shape[-1]
                    and targets[token_idx] != IMAGE_TOKEN_INDEX
                ):
                    targets[token_idx] = IGNORE_INDEX
                    token_idx += 1
            # do not train on padding tokens
            targets[targets == pad_token] = IGNORE_INDEX

        # mask image tokens is unnecessary for llava-1.5
        # targets[targets == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        # print(input_ids.shape)

        return dict(input_ids=input_ids, labels=targets, image=images)
