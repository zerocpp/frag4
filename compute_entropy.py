'''
计算
'''

import argparse
import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
from core.computation.uncertainty_measure import cluster_assignment_entropy
from core.data.data_utils import load_ds_from_json

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# 计算准确率
def compute_accuracy(scores):
    if len(scores) == 0:
        return -1
    return sum(1 for score in scores if score >= 0.5) / len(scores)

# 计算语义熵
def compute_entropy(cluster_ids):
    if len(cluster_ids) == 0:
        return -1
    return cluster_assignment_entropy(cluster_ids)

def get_parser():
    # python compute_entropy.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output/train/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--output_dir", type=str, default="output/result/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--output_filename", type=str, default="entroy.pkl")
    parser.add_argument("--dataset_json_file", type=str, default=None)
    return parser

def main(args):
    assert os.path.exists(args.input_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    assert args.dataset_json_file, f"Dataset json file is required. Got {args.dataset_json_file}"
    id_list, data_dict = load_ds_from_json(args.dataset_json_file)

    entropy_results = defaultdict(dict)
    for example_id in tqdm(id_list, desc="Computing Entropy"):
        assert example_id in data_dict, f"Example id {example_id} not found in dataset {args.dataset_json_file}."
        example = data_dict[example_id]

        input_path = os.path.join(args.input_dir, f"{example_id}.pkl")
        if not os.path.exists(input_path): # 若输入文件不存在
            continue
        input_dict = load_pickle_file(input_path)

        entropy_results[example_id]['semantic_entropy'] = compute_entropy(input_dict['cluster_ids'])

    # 保存结果
    output_path = os.path.join(args.output_dir, args.output_filename)
    save_pickle_file(output_path, entropy_results)

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    print(f'args: {args}')
    main(args)
    print('Done')
