"""
Predict with LLM on task.
"""

import gc
import os
import logging
import random
from tqdm import tqdm
import argparse
import numpy as np
import torch
import pickle

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from core.data.data_utils import load_ds, load_ds_from_json
from core.utils import utils
from core.models.huggingface_models import HuggingfaceModel

# 低于该温度视为贪婪采样
GREEDY_TEMPERATURE_THRESHOLD = 0.11

def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.FATAL)  # logging.DEBUG

def make_brief_prompt(question, context=None):
    '''构造简短回答的prompt'''
    # 简短回答的指令
    prompt = "Answer the following question as briefly as possible.\n"
    if context:
        prompt += f"Context: {context}"
    prompt += f"Question: {question}"
    prompt += f"Answer:"
    return prompt

def get_output_path(args, example_id):
    '''返回输出文件路径'''
    # `./output/train/generation/Qwen/Qwen2.5-1.5B-Instruct/squad/greedy_golden/1.pkl`
    # sample = "greedy" if args.temperature < GREEDY_TEMPERATURE_THRESHOLD else "sample"
    # context = ""
    # if args.use_context:
    #     if args.irrelevant_context:
    #         context = "irrelevant"
    #     else:
    #         context = "golden"
    # else:
    #     context = "without"
    # sample_context = f"{sample}_{context}"
    dir_path = args.output_dir
    # f"{args.output_dir}/{args.split}/generation/{args.model}/{args.dataset}/{sample_context}"
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
            'responses': [],
        }
    result = load_pickle_file(file_path)
    return result


def main(args):
    # 加载模型
    model = HuggingfaceModel(args.model, stop_sequences='default', max_new_tokens=args.max_new_tokens)

    # 创建输出目录
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    assert args.dataset_json_file, f"Dataset json file is required. Got {args.dataset_json_file}"
    id_list, data_dict = load_ds_from_json(args.dataset_json_file)

    # 根据prompt生成回答
    def generate_responses(prompt):
        last_error = None
        for _ in range(args.retry_times):
            output = model.predict(prompt, temperature=args.temperature, return_latent=args.return_latent)
            if 'error' not in output:
                return output
            last_error = output['error']
        if last_error is not None:
            return {'error': last_error}

    for i, example_id in enumerate(tqdm(id_list, desc="Generating")):
        # 释放显存
        if i % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        assert example_id in data_dict, f"Example id {example_id} not found in dataset {args.dataset_json_file}."
        example = data_dict[example_id]
        output_path = os.path.join(args.output_dir, f"{example_id}.pkl")
        result = get_result(example_id, args)
        num_gen = max(0, args.num_generations - len(result['responses'])) # 还需要生成的数量
        if args.override: # 覆盖
            result['responses'] = []
            num_gen = args.num_generations
        if not args.override and num_gen <= 0: # 若 不覆盖 且 生成的回答已足够
            continue

        # 构造prompt
        if args.use_context:
            if args.irrelevant_context:
                prompt = make_brief_prompt(example['question'], example['irrelevant_context'])
            else:
                prompt = make_brief_prompt(example['question'], example['context'])
        else:
            prompt = make_brief_prompt(example['question'], None)

        # 生成回答
        for _ in range(num_gen):
            response = generate_responses(prompt)
            if 'error' not in response:
                result['responses'].append(response)

        # 保存结果
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)

def get_parser():
    '''
    python generate_responses.py --output_dir output/train/generation/Qwen/Qwen2.5-7B-Instruct/squad/sample_golden --model Qwen/Qwen2.5-7B-Instruct --num_generations 10 --retry_times 3 --temperature 1.0 --max_new_tokens 50 --use_context --irrelevant_context --return_latent --dataset_json_file output/dataset/squad_train.json --override
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output/train/generation/Qwen/Qwen2.5-7B-Instruct/squad/sample_golden')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--retry_times', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--use_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--irrelevant_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--return_latent", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_json_file", type=str, default=None)
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
    return parser

if __name__ == '__main__':
    setup_logger()
    parser = get_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        raise ValueError(f'Unkown args: {unknown}')
    print(f"args: {args}")
    main(args)
    print("Done!")

