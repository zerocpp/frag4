"""Data Loading Utilities."""
import logging
import os
import json
import hashlib
import datasets

import os
import datasets
import json
from tqdm import tqdm

# 镜像加速
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

DATASET_ROOT = "/Users/song/datasets"
DATASET_SQUAD = "rajpurkar/squad_v2"

# 数据集绝对路径
DATASET_PATH = os.path.join(DATASET_ROOT, DATASET_SQUAD)

# 从huggingface拉取数据集并加载
ds = datasets.load_dataset(DATASET_SQUAD)
print(ds)

# 将数据集保存到本地
ds.save_to_disk(DATASET_PATH)

# 从本地加载数据集
ds = datasets.load_from_disk(DATASET_PATH)
print(ds)
