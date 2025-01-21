import warnings
import os
from ..path import default_dataset_dir


num_shards = 23099
download_mmc4 = False
download_mmc4_core = False

mmc4_save_dir = os.path.join(default_dataset_dir, "mmc4_download")
mmc4_core_save_dir = os.path.join(default_dataset_dir, "mmc4_core_download")

if download_mmc4:
    if os.path.exists(mmc4_save_dir):
        warnings.warn(f"{mmc4_save_dir} already exists. Skipping download.")
    else:
        os.makedirs(mmc4_save_dir)
        for i in range(num_shards):
            os.system(f"wget -P {mmc4_save_dir} https://storage.googleapis.com/ai2-jackh-mmc4-gated-public-41423/data_v1.1/docs_shard_{i}_v2.jsonl.zip")
            os.system(f"unzip {mmc4_save_dir}/docs_shard_{i}_v2.jsonl.zip -d {mmc4_save_dir}")

if download_mmc4_core:
    if os.path.exists(mmc4_core_save_dir):
        warnings.warn(f"{mmc4_core_save_dir} already exists. Skipping download.")
    else:
        os.makedirs(mmc4_core_save_dir)
        for i in range(num_shards):
            os.system(f"wget -P {mmc4_core_save_dir} https://storage.googleapis.com/ai2-jackh-mmc4-gated-public-41423/data_core_v1.1/docs_shard_{i}_v3.jsonl.zip")
            os.system(f"unzip {mmc4_core_save_dir}/docs_shard_{i}_v3.jsonl.zip -d {mmc4_core_save_dir}")
