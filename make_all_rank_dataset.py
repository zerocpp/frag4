import os
import json
import jsonlines
import pandas as pd
from collections import defaultdict
import shutil
from tqdm import tqdm
import csv
from make_rank_dataset import load_dataset, load_rank_results, make_rank_dataset

# 数据集目录
DATASET_DIR = '/home/song/dataset/beir'
# 检索结果目录
RANK_RESULT_DIR = '/home/song/dataset/first/beir_rank'
# FIRST所使用的11个数据集
BEIR_DATASET_NAMES = ["trec-covid", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "msmarco",  "nfcorpus", "nq", "scidocs", "scifact"]
# 尺寸后缀、问题数量、文档数量
SIZES = [("toy", 10, 10), ("small", 50, 10)]

for dataset_name in tqdm(BEIR_DATASET_NAMES, desc="Building rank datasets"):
    dataset_path = os.path.join(DATASET_DIR, dataset_name)
    rank_result_path = os.path.join(RANK_RESULT_DIR, dataset_name, "rank.tsv")
    for size_name, query_num, rank_num in SIZES:
        output_path = os.path.join('dataset/rank', dataset_name, f"{dataset_name}-{size_name}.jsonl")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        make_rank_dataset(dataset_name, dataset_path, rank_result_path, output_path, query_num, rank_num)

print("Done!")
