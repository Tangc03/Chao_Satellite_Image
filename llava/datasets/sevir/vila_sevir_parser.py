from typing import Dict, Any, Union
import datetime
import numpy as np
from PIL import Image

from .sevir_dataloader import (
    SEVIRDataLoader,
    SEVIR_RAW_SEQ_LEN,
    SEVIR_CATALOG,
    SEVIR_DATA_DIR,
)


class SEVIRVILAParser():

    def __init__(self,
                 cfg: Dict[str, Any],
                 in_len: int = None,
                 ):
        start_date = datetime.datetime(*cfg["start_date"]) \
            if cfg["start_date"] is not None else None
        end_date = datetime.datetime(*cfg["end_date"]) \
            if cfg["end_date"] is not None else None

        self.frame_stride = cfg["frame_stride"]
        self.sevir_dataloader = SEVIRDataLoader(
            data_types=["vil", ],
            seq_len=cfg["seq_len"]*self.frame_stride,
            raw_seq_len=SEVIR_RAW_SEQ_LEN,
            sample_mode="sequent",
            stride=cfg["stride"]*self.frame_stride,
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
            shuffle=cfg["shuffle"],
            shuffle_seed=cfg["shuffle_seed"],
            output_type=np.float32,
            preprocess=True,
            rescale_method="01",
            downsample_dict=None,
            verbose=True,
        )
        self.in_len = in_len

    def parse(self,
              idx: Union[str, int],
              ret_PIL: bool = False,
              ):
        r"""

        Parameters
        ----------
        idx : Union[str, int]
        ret_PIL : bool

        Returns
        -------

        """
        if isinstance(idx, str):
            idx = int(idx)
        data_dict = self.sevir_dataloader[idx]
        seq = data_dict["vil"].squeeze(0)[..., ::self.frame_stride]
        seq = seq.float().cpu().numpy()
        if self.in_len is not None:
            seq = seq[..., :self.in_len]
        if ret_PIL:
            ret = self.seq_to_img_list(seq)
        else:
            ret = seq
        return ret

    def seq_to_img_list(self, seq: np.ndarray):
        r"""

        Parameters
        ----------
        seq: np.ndarray
        shape = (h w t)

        Returns
        -------

        """
        img_list = []
        for i in range(seq.shape[-1]):
            img = Image.fromarray(seq[..., i])
            img_list.append(img)
        return img_list
