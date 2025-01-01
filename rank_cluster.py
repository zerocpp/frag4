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

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True, help="jsonl file path")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing generated pickle files")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save clustered pickle files")
    parser.add_argument("--model", type=str, default="microsoft/deberta-v2-xxlarge-mnli")
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
    return parser

def get_semantic_ids(strings_list, model, question, strict_entailment=False):
    """Group list of predictions into semantic meaning."""
    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, question)
        implication_2 = model.check_implication(text2, text1, question)  # pylint: disable=arguments-out-of-order
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
    assert os.path.exists(args.input_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    assert args.dataset_path, "Dataset path is required."
    assert args.dataset_path.endswith('.jsonl'), "Dataset path must be a jsonl file."
    def get_task_total_count():
        '''获取任务总数'''
        with jsonlines.open(args.dataset_path) as reader:
            return len(list(reader))
    total_count = get_task_total_count()

    # 返回生成结果
    def load_gen_result(id_):
        file_path = os.path.join(args.input_dir, f"{id_}.pkl")
        if os.path.exists(file_path):
            return load_pickle_file(file_path)
        return None
    
    # 返回聚类结果
    def load_cluster_result(id_):
        file_path = os.path.join(args.output_dir, f"{id_}.pkl")
        if os.path.exists(file_path):
            return load_pickle_file(file_path)
        return None
    
    # 保存结果
    def save_cluster_result(result):
        file_path = os.path.join(args.output_dir, f"{result['id']}.pkl")
        # 创建文件夹
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

    # 加载NLI模型
    model = EntailmentDeberta(args.model)

    with jsonlines.open(args.dataset_path) as reader:
        for i, item in enumerate(tqdm(reader, desc="RankCluster", total=total_count)):
            # 释放显存
            if i % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()

            id_ = item['id']
            gen_result = load_gen_result(id_)
            if gen_result is None:
                continue # 若生成结果为空，则跳过
            if len(gen_result['sample']) == 0:
                continue # 若生成结果为空，则跳过

            cluster_result = load_cluster_result(id_)
            execute = args.override
            if cluster_result is None: # 若聚类结果为空
                cluster_result = gen_result
                execute = True
            else:
                num_responses = len(gen_result['sample'])
                num_cluster_ids = len(cluster_result['cluster_ids'])
                if num_responses != num_cluster_ids: # 若生成数和聚类数不相等
                    cluster_result = gen_result
                    execute = True
            
            if not execute:
                continue # 若不需要执行，则跳过
            
            # 聚类
            cluster_ids = get_semantic_ids(cluster_result['sample'], 
                                       model=model, 
                                       question=item['question'],
                                       strict_entailment=False)
            cluster_result['cluster_ids'] = cluster_ids

            # 保存聚类结果
            save_cluster_result(cluster_result)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('> Args:', args)
    main(args)
    print('> Done!')
