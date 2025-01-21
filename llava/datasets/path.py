import os

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

default_exps_dir = os.path.abspath(os.path.join(root_dir, "checkpoints"))

default_dataset_dir = os.path.abspath(os.path.join(root_dir, "playground", "data"))
# SEVIR
default_dataset_sevir_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevir"))
default_dataset_sevirlr_dir = os.path.abspath(os.path.join(default_dataset_dir, "sevirlr"))
