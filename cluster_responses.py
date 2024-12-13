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
    # python cluster_responses --input_dir output/train/generation/Qwen/Qwen2.5-7B-Instruct/squad/sample_golden --output_dir output/train/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --model microsoft/deberta-v2-xxlarge-mnli --no-override --dataset_json_file output/dataset/squad_train.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output/train/generation/Qwen/Qwen2.5-7B-Instruct/squad/sample_golden")
    parser.add_argument("--output_dir", type=str, default="output/train/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v2-xxlarge-mnli")
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_json_file", type=str, default=None)
    return parser

def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""
    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example["question"])
        implication_2 = model.check_implication(text2, text1, example["question"])  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)
        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids

def get_output_path(args, example_id):
    '''返回输出文件路径'''
    # `./output/train/clustered/Qwen/Qwen2.5-1.5B-Instruct/squad/sample_golden/1.pkl`
    dir_path = args.output_dir
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.join(dir_path, f"{example_id}.pkl")

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def get_result(example_id, args):
    file_path = get_output_path(args, example_id)
    if not os.path.exists(file_path):
        return {
            'example_id': example_id,
            'cluster_ids': [],
        }
    result = load_pickle_file(file_path)
    return result


def main(args):
    assert os.path.exists(args.input_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    assert args.dataset_json_file, f"Dataset json file is required. Got {args.dataset_json_file}"
    id_list, data_dict = load_ds_from_json(args.dataset_json_file)

    # 加载NLI模型
    model = EntailmentDeberta(args.model)

    for example_id in tqdm(id_list, desc="Clustering"):
        assert example_id in data_dict, f"Example id {example_id} not found in dataset {args.dataset_json_file}."
        example = data_dict[example_id]

        input_path = os.path.join(args.input_dir, f"{example_id}.pkl")
        if not os.path.exists(input_path): # 若输入文件不存在
            continue
        responses = load_pickle_file(input_path)['responses']
        num_responses = len(responses)

        output_path = os.path.join(args.output_dir, f"{example_id}.pkl")

        override = args.override
        if not override and os.path.exists(output_path): # 若不强制覆盖且输出文件存在
            cluster_ids = load_pickle_file(output_path)['cluster_ids']
            num_cluster_ids = len(cluster_ids)
            override = num_responses != num_cluster_ids # 若生成数和聚类数不相等，则覆盖
        
        if not override:
            continue
        
        # 生成文本
        response_texts = [resp.get('text', '') for resp in responses]
        # 聚类
        cluster_ids = get_semantic_ids(response_texts, 
                                       model=model, 
                                       strict_entailment=False, 
                                       example=example)
        result = {
            'example_id': example_id,
            'cluster_ids': cluster_ids,
        }
        save_pickle_file(output_path, result)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    main(args)
    print('cluster_responses done')
