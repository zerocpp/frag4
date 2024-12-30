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
# SIZES = [("toy", 10, 10), ("small", 50, 10)]
# SIZES = [("small100", 50, 100), ("large", 500, 10)]
# SIZES = [("all", 10000, 10)]
SIZES = [("large", 500, 10)]

num_lines = []
for dataset_name in tqdm(BEIR_DATASET_NAMES, desc="Building rank datasets"):
    for size_name, query_num, rank_num in SIZES:
        output_path = os.path.join('dataset/rank', dataset_name, f"{dataset_name}-{size_name}.jsonl")
        # 查看文件行数
        with open(output_path, 'r') as f:
            num_lines.append({'dataset': dataset_name, 'line': sum(1 for line in f)})
        # os.makedirs(os.path.dirname(output_path), exist_ok=True)
        # make_rank_dataset(dataset_name, dataset_path, rank_result_path, output_path, query_num, rank_num)
print(num_lines)
# 求总数
print('total:', sum([x['line'] for x in num_lines]))
# print("Done!")


# output_path = os.path.join('dataset/rank', dataset_name, f"{dataset_name}-{size_name}.jsonl")