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

from core.data.data_utils import load_ds
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
    sample = "greedy" if args.temperature < GREEDY_TEMPERATURE_THRESHOLD else "sample"
    context = ""
    if args.use_context:
        if args.irrelevant_context:
            context = "irrelevant"
        else:
            context = "golden"
    else:
        context = "without"
    sample_context = f"{sample}_{context}"
    dir_path = f"{args.output_dir}/{args.split}/generation/{args.model}/{args.dataset}/{sample_context}"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return os.path.join(dir_path, f"{example_id}.pkl")


def main(args):
    dataset = load_ds(args.dataset, args.split)
    model = HuggingfaceModel(args.model, stop_sequences='default', max_new_tokens=args.max_new_tokens)
    # model.predict(input_data, temperature, return_full=False, return_latent=False)
    
    num_samples = args.num_samples
    if num_samples is None or num_samples < 0 or num_samples > len(dataset):
        num_samples = len(dataset)
    logging.info(f"Generating responses for {num_samples} samples")

    for i in tqdm(range(num_samples)):
    # for i, example in tqdm(enumerate(dataset_split)):
        # 释放显存
        gc.collect()
        torch.cuda.empty_cache()

        example = dataset[i]
        # load result
        file_path = get_output_path(args, example['id'])
        if os.path.exists(file_path):
            continue # 如果结果已经存在，则跳过
        
        if args.use_context:
            if args.irrelevant_context:
                prompt = make_brief_prompt(example['question'], example['irrelevant_context'])
            else:
                prompt = make_brief_prompt(example['question'], example['context'])
        else:
            prompt = make_brief_prompt(example['question'], None)
        
        def generate_responses(prompt):
            last_error = None
            for _ in range(args.retry_times):
                output = model.predict(prompt, temperature=args.temperature, return_latent=args.return_latent)
                if 'error' not in output:
                    return output
                last_error = output['error']
            if last_error is not None:
                return {'error': last_error}

        result = {
            'example_id': example['id'],
            'args': args,
            'responses': [],
        }

        for _ in range(args.num_generations):
            response = generate_responses(prompt)
            result['responses'].append(response)

        # save result
        with open(file_path, 'wb') as f:
            pickle.dump(result, f)

def get_parser():
    '''
    python generate_responses.py --dataset squad --split train --model Qwen/Qwen2.5-7B-Instruct --num_samples 2000 --num_generations 10 --temperature 1.0 --use_context --irrelevant_context --return_latent
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='output')
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-1.5B-Instruct')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument('--num_generations', type=int, default=10)
    parser.add_argument('--retry_times', type=int, default=3)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--use_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--irrelevant_context", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--return_latent", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_json_file", type=str, default=None)
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

