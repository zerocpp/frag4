'''
根据语义熵进行重排序
	读取dataset-jsonl
	读取对应的cluster
	判断cluster_ids个数，并计算语义熵
	按语义熵升序进行重排序
	得到tsv文件
'''

import gc
import os
import pickle
import jsonlines
import torch
from tqdm import tqdm
from collections import defaultdict
import argparse
from core.models.entailment import EntailmentDeberta
from core.data.data_utils import load_ds_from_json
from core.computation.uncertainty_measure import cluster_assignment_entropy

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_cluster_ids(clustered_file_path):
    cluster_ids = []
    if os.path.exists(clustered_file_path):
        res = load_pickle_file(clustered_file_path)
        cluster_ids = res.get('cluster_ids', [])
    return cluster_ids

# 计算语义熵
def compute_entropy(cluster_ids):
    if len(cluster_ids) == 0:
        return -1
    return cluster_assignment_entropy(cluster_ids)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="jsonl file path")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing clustered pickle files")
    parser.add_argument("--output_path", type=str, required=True, help="rerank.tsv Path to save the rerank result")
    return parser

def valid_args(dataset_path, input_dir, output_path):
    assert os.path.exists(input_dir)

    assert output_path.endswith('.tsv'), "Output path must be a tsv file."
    # 创建输出目录
    output_dir = os.path.dirname(output_path)
    print(f'Output dir: {output_dir}')
    os.makedirs(output_dir, exist_ok=True)

    assert dataset_path, "Dataset path is required."
    assert dataset_path.endswith('.jsonl'), "Dataset path must be a jsonl file."

def rerank_by_entropy(dataset_path, input_dir, output_path):
    # 验证参数
    valid_args(dataset_path, input_dir, output_path)

    # 加载数据集
    def get_task_total_count():
        '''获取任务总数'''
        with jsonlines.open(dataset_path) as reader:
            return len(list(reader))
    total_count = get_task_total_count()
    
    # 返回聚类结果
    def load_cluster_ids(id_):
        file_path = os.path.join(input_dir, f"{id_}.pkl")
        if os.path.exists(file_path):
            result = load_pickle_file(file_path)
            return result.get('cluster_ids', [])
        return []

    rerank_results = []
    with jsonlines.open(dataset_path) as reader:
        for i, item in enumerate(tqdm(reader, desc="Rerank", total=total_count)):
            id_ = item['id']
            doc_id = item['metadata']['doc_id']
            if doc_id == 'no':
                continue # 重排序的时候，跳过no的doc_id
            cluster_ids = load_cluster_ids(id_)
            if len(cluster_ids) == 0:
                continue
            entropy = compute_entropy(cluster_ids)
            score = entropy
            rerank_results.append({
                'query_id': item['metadata']['query_id'],
                'doc_id': item['metadata']['doc_id'],
                'score': score,
            })
    # 先按query_id升序排序，再按score降序排序
    rerank_results = sorted(rerank_results, key=lambda x: (x['query_id'], x['score']))
    # 保存成tsv文件
    with open(output_path, 'w') as f:
        for item in rerank_results:
            f.write(f"{item['query_id']}\t{item['doc_id']}\t{item['score']}\n")

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('> Args:', args)
    rerank_by_entropy(args.dataset_path, args.input_dir, args.output_path)
    print('> Done!')
