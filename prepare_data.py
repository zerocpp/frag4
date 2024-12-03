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
    # 如果file_path的目录不存在，则创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_parser():
    # python prepare_data.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    return parser

def load_responses(generation_file_path):
    responses = []
    if os.path.exists(generation_file_path):
        res = load_pickle_file(generation_file_path)
        # d['greedy_golden']['responses'][0]['hidden_states']['sec_last_token_embedding']
        for r in res['responses']:
            if 'error' in r:
                continue
            it = {}
            if 'text' in r:
                it['text'] = r['text']
            if 'hidden_states' in r:
                it['slt_embedding'] = r['hidden_states']['sec_last_token_embedding']
            responses.append(it)
    return responses

def load_cluster_ids(clustered_file_path):
    cluster_ids = []
    if os.path.exists(clustered_file_path):
        res = load_pickle_file(clustered_file_path)
        cluster_ids = res.get('cluster_ids', [])
    return cluster_ids

def merge_responses_and_cluster_ids(responses, cluster_ids):
    for i, r in enumerate(responses):
        if len(cluster_ids) == len(responses):
            r['cluster_id'] = cluster_ids[i]
        else:
            r['cluster_id'] = -1
    return responses

def prepare_data(dataset_name, split, model_name):
    model_short_name = model_name.split("/")[0]
    # 加载数据集
    dataset_json_file = os.path.join(args.root_dir, f"output/dataset/{dataset_name}_{split}.json")
    assert os.path.exists(dataset_json_file), f"Dataset json file {dataset_json_path} not found."
    id_list, data_dict = load_ds_from_json(dataset_json_file)

    result = {
        'id': id_list,
        'data': defaultdict(dict),
    }
    for i, example_id in enumerate(tqdm(id_list, desc=f"Preparing {dataset_name} {split} {model_short_name}")):
        assert example_id in data_dict, f"Example id {example_id} not found in dataset {args.dataset_json_file}."
        example = data_dict[example_id]
        example_result = defaultdict(dict)
        example_result['example'] = example

        for sample_suffix in ["golden", "irrelevant", "without"]:
            example_result[sample_suffix] = {}

            # greedy
            generation_file_path = os.path.join(args.root_dir, f"output/{split}/generation/{model_name}/{dataset_name}/greedy_{sample_suffix}/{example_id}.pkl")
            greedy_responses = load_responses(generation_file_path)
            if len(greedy_responses) == 1:
                example_result[sample_suffix]['greedy'] = greedy_responses[0]
            else:
                example_result[sample_suffix]['greedy'] = {}
            
            # sample
            generation_file_path = os.path.join(args.root_dir, f"output/{split}/generation/{model_name}/{dataset_name}/sample_{sample_suffix}/{example_id}.pkl")
            sample_responses = load_responses(generation_file_path)
            clustered_file_path = os.path.join(args.root_dir, f"output/{split}/clustered/{model_name}/{dataset_name}/sample_{sample_suffix}/{example_id}.pkl")
            cluster_ids = load_cluster_ids(clustered_file_path)
            sample_responses = merge_responses_and_cluster_ids(sample_responses, cluster_ids)
            example_result[sample_suffix]['sample'] = sample_responses

        result['data'][example_id] = example_result

    return result

def main(args):
    for dataset_name in ["squad", "triviaqa"]:
        for split in ["train", "validation"]:
            for model_name in ["Qwen/Qwen2.5-7B-Instruct", "meta-llama/Llama-3.1-8B-Instruct"]:
                result = prepare_data(dataset_name, split, model_name)
                output_path = os.path.join(args.root_dir, "output/result", model_name, f"{dataset_name}_{split}.pkl")
                save_pickle_file(output_path, result)


if __name__ == '__main__':
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    print(f"args: {args}")
    main(args)
    print("Done!")

