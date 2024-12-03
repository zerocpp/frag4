'''
数据准备脚本
1. 汇总下列数据到一起：
- example
- greedy generation
- greedy hidden states
- high-t generations
- cluster_ids
2. 计算语义熵
'''

import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import argparse
from core.models.entailment import EntailmentDeberta
from core.data.data_utils import load_ds_from_json

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_parser():
    # python prepare_data.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_json_file", type=str, default="output/dataset/squad_train.json")
    parser.add_argument("--output_path", type=str, default="output/dataset/squad_train.pkl")
    return parser

def main(args):
    pass


if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    print(f"args: {args}")
    main(args)
    print("Done!")

