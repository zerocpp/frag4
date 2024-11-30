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

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)
    
def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

# 计算精度
def compute_accuracy(scores):
    if len(scores) == 0:
        return -1
    return sum(1 for score in scores if score >= 0.5) / len(scores)

# 计算熵
def compute_entropy(cluster_ids):
    if len(cluster_ids) == 0:
        return -1
    return cluster_assignment_entropy(cluster_ids)

# def compute_dataset(dataset_name):
#     computed_results = defaultdict(dict)
#     for with_or_without_context in ["with_context", "without_context"]:
#         dir_path = f"output/llama318i/{dataset_name}/{with_or_without_context}"
#         for file_name in tqdm(os.listdir(dir_path), desc=f"Loading {dataset_name} {with_or_without_context}"):
#             file_path = os.path.join(dir_path, file_name)
#             result = load_pickle_file(file_path)
#             example = result['example']
#             example_id = example['id']
#             if example_id not in computed_results:
#                 computed_results[example_id] = {
#                     'question': example['question'],
#                     'answers': example['answers'],
#                     'context': example['context'],
#                 }
#             computed_results[example_id][with_or_without_context] = {
#                 'responses': [it['sliced_answer'] for it in result['responses']],
#                 'scores_llama70': result.get('scores_by_o_llama70', []),
#                 'scores_qwen72': result.get('scores_by_o_qwen72', []),
#                 'accuracy_llama70': compute_accuracy(result.get('scores_by_o_llama70', [])),
#                 'accuracy_qwen72': compute_accuracy(result.get('scores_by_o_qwen72', [])),
#                 'cluster_ids_deberta': result.get('cluster_ids_no_strict', []),
#                 'semantic_entropy_deberta': compute_entropy(result.get('cluster_ids_no_strict', [])),
#                 'cluster_ids_llama70': result.get('cluster_ids_o_llama70', []),
#                 'semantic_entropy_llama70': compute_entropy(result.get('cluster_ids_o_llama70', [])),
#                 'cluster_ids_qwen72': result.get('cluster_ids_o_qwen72', []),
#                 'semantic_entropy_qwen72': compute_entropy(result.get('cluster_ids_o_qwen72', [])),
#             }
#     return computed_results

# triviaqa_computed_results = compute_dataset('triviaqa')
# save_pickle_file('output/triviaqa_computed_results.pkl', triviaqa_computed_results)
# squad_computed_results = compute_dataset('squad')
# save_pickle_file('output/squad_computed_results.pkl', squad_computed_results)
# print("Done")


def get_parser():
    # python compute_entropy.py --input_dir output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --output_dir output/result/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--output_dir", type=str, default="output/result/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    return parser

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    for file_name in os.listdir(args.input_dir):
        input_path = os.path.join(args.input_dir, file_name)
        data = load_pickle_file(input_path)
        cluster_ids = data['cluster_ids']
        semantic_entropy = compute_entropy(cluster_ids)

        output_path = os.path.join(args.output_dir, file_name)
        if os.path.exists(output_path):
            result = load_pickle_file(output_path)    
        else:
            result = {
                'example_id': data['example_id'],
            }
        result['semantic_entropy'] = semantic_entropy
        save_pickle_file(output_path, result)

if __name__ == "__main__":
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    print(f'args: {args}')
    main(args)
    print('Done')
