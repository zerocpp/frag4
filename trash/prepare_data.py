'''
数据准备脚本
1. 汇总下列数据到一起：
- example
- greedy generation
- greedy hidden states
- high-t generations
- cluster_ids
2. 计算语义熵

最终格式like:
{
	"example": {
		"id": "id1",
		"question": "q1",
		"context": "c1",
		"answers": ["a1"]
	},
	"golden": {
		"greedy": {
			"text": "",
			"tbg_emb": Tensor(),
			"slt_emb": Tensor(),
            "accuracy": 1.0,
		},
		"sample": ["r1","r2"],
		"cluster_ids": [0,1],
        "accuracy": [1.0, 0.0],
		"entropy": 1.5
	},
	"irrelevant": {
		"greedy": {
			"text": "",
			"tbg_emb": Tensor(),
			"slt_emb": Tensor(),
            "accuracy": 1.0,
		},
		"sample": ["r1","r2"],
		"cluster_ids": [0,1],
        "accuracy": [1.0, 0.0],
		"entropy": 1.5
	},
	"without": {
		"greedy": {
			"text": "",
			"tbg_emb": Tensor(),
			"slt_emb": Tensor(),
            "accuracy": 1.0,
		},
		"sample": ["r1","r2"],
		"cluster_ids": [0,1],
        "accuracy": [1.0, 0.0],
		"entropy": 1.5
	},
}
'''

import os
import pickle
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
    # 如果file_path的目录不存在，则创建
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_parser():
    # python prepare_data.py
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    return parser

def load_text_responses(generation_file_path):
    responses = []
    if os.path.exists(generation_file_path):
        res = load_pickle_file(generation_file_path)
        for r in res['responses']:
            if 'error' in r:
                continue
            responses.append(r['text'])
    return responses

def load_responses(generation_file_path):
    responses = []
    if os.path.exists(generation_file_path):
        res = load_pickle_file(generation_file_path)
        for r in res['responses']:
            if 'error' in r:
                continue
            it = {}
            it['text'] = r['text']
            it['slt_emb'] = r['hidden_states']['sec_last_token_embedding']
            it['tbg_emb'] = r['hidden_states']['last_tok_bef_gen_embedding']
            # if 'text' in r:
            #     it['text'] = r['text']
            # if 'hidden_states' in r:
            #     it['slt_emb'] = r['hidden_states']['sec_last_token_embedding']
            #     it['tbg_emb'] = r['hidden_states']['last_tok_bef_gen_embedding']
            responses.append(it)
    return responses

def load_greedy_response(generation_file_path):
    responses = load_responses(generation_file_path)
    assert len(responses) <= 1, f"More than one response found in {generation_file_path}."
    return responses[0] if responses else None

def load_cluster_ids(clustered_file_path):
    cluster_ids = []
    if os.path.exists(clustered_file_path):
        res = load_pickle_file(clustered_file_path)
        cluster_ids = res.get('cluster_ids', [])
    return cluster_ids

def load_accuracy(eval_file_path):
    accuracy = []
    if os.path.exists(eval_file_path):
        res = load_pickle_file(eval_file_path)
        accuracy = res.get('qwen_scores', [])
    return accuracy

def load_greedy_accuracy(eval_file_path):
    accuracy = load_accuracy(eval_file_path)
    assert len(accuracy) <= 1, f"More than one accuracy found in {eval_file_path}."
    return accuracy[0] if accuracy else None

def get_dataset_json_filename(dataset_name, split, size='small'):
    assert size in ['small', 'large'], f"Invalid size {size}."
    size_map = {
        'small': {
            'train': 2000,
            'validation': 100,
            'test': 100,
        },
        'large': {
            'train': 10000,
            'validation': 1000,
            'test': 1000,
        }
    }
    num_samples = size_map[size][split]
    return f"{dataset_name}_{split}_{num_samples}.json"

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

def prepare_data(dataset_name, split, model_name):
    model_short_name = model_name.split("/")[0]
    # 加载数据集
    dataset_json_file = os.path.join(args.root_dir, f"output/dataset/{get_dataset_json_filename(dataset_name, split)}")
    assert os.path.exists(dataset_json_file), f"Dataset json file {dataset_json_file} not found."
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
            greedy_gen_path = os.path.join(args.root_dir, f"output/{split}/generation/{model_name}/{dataset_name}/greedy_{sample_suffix}/{example_id}.pkl")
            greedy_dict = load_greedy_response(greedy_gen_path)
            example_result[sample_suffix]['greedy'] = greedy_dict

            # greedy accuracy
            # 如果greedy_dict为dict，则计算accuracy
            if type(greedy_dict) == dict:
                eval_path = os.path.join(args.root_dir, f"output/{split}/evaluation/{model_name}/{dataset_name}/greedy_{sample_suffix}/{example_id}.pkl")
                example_result[sample_suffix]['greedy']['accuracy'] = load_greedy_accuracy(eval_path)
            
            # sample
            sample_gen_path = os.path.join(args.root_dir, f"output/{split}/generation/{model_name}/{dataset_name}/sample_{sample_suffix}/{example_id}.pkl")
            example_result[sample_suffix]['sample'] = load_text_responses(sample_gen_path)

            # sample accuracy
            eval_path = os.path.join(args.root_dir, f"output/{split}/evaluation/{model_name}/{dataset_name}/sample_{sample_suffix}/{example_id}.pkl")
            example_result[sample_suffix]['accuracy'] = load_accuracy(eval_path)

            # cluster_ids
            cluster_path = os.path.join(args.root_dir, f"output/{split}/clustered/{model_name}/{dataset_name}/sample_{sample_suffix}/{example_id}.pkl")
            cluster_ids = load_cluster_ids(cluster_path)
            example_result[sample_suffix]['cluster_ids'] = cluster_ids

            # entropy
            example_result[sample_suffix]['entropy'] = compute_entropy(cluster_ids)

        result['data'][example_id] = example_result

    return result

def main(args):
    for dataset_name in ["squad", "triviaqa"]:
        for split in ["train", "validation", "test"]:
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

