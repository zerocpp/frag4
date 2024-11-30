import os
import pickle
from tqdm import tqdm
from collections import defaultdict
import argparse
from core.models.entailment import EntailmentDeberta
from core.data.data_utils import load_ds

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_parser():
    # python cluster_responses --model microsoft/deberta-v2-xxlarge-mnli --dataset squad --split train --num_samples 2000 --input_dir output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --output_dir output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--output_dir", type=str, default="output/clustered/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v2-xxlarge-mnli")
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
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

def main(args):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    dataset = load_ds(args.dataset, args.split)
    data_dict = defaultdict(dict)
    for i in tqdm(range(args.num_samples)):
        data = dataset[i]
        data_dict[data['id']] = data

    # 加载NLI模型
    model = EntailmentDeberta(args.model)

    input_dir = args.input_dir
    assert os.path.exists(input_dir)

    for file in tqdm(os.listdir(input_dir), desc="Clustering"):
        # 判断结果是否已存在
        if not args.override: # 若不覆盖
            output_path = os.path.join(args.output_dir, file)
            if os.path.exists(output_path): # 若结果已存在
                continue # 跳过

        # 加载生成文本
        file_path = os.path.join(input_dir, file)
        response_dict = load_pickle_file(file_path)
        example_id = response_dict['example_id']
        assert example_id in data_dict
        example = data_dict[example_id]
        response_texts = [resp.get('text', '') for resp in response_dict['responses']]
        
        # 聚类
        cluster_ids = get_semantic_ids(response_texts, 
                                       model=model, 
                                       strict_entailment=False, 
                                       example=example)
        
        # 保存结果
        result = {
            'example_id': example_id,
            'args': args,
            'question': example['question'],
            'responses': response_dict['responses'],
            'cluster_ids': cluster_ids,
        }
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        save_pickle_file(os.path.join(output_dir, file), result)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    main(args)
    print('cluster_responses done')
