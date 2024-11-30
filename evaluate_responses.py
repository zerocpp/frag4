import gc
import logging
import random
from tqdm import tqdm
from transformers import StoppingCriteria
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteriaList
from core.data.data_utils import load_ds

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import argparse
import pickle

import logging
import os
import json
import hashlib
import datasets

from core.models.llm import OllamaLLM, BaseLLM

# 忽略警告
import warnings
warnings.filterwarnings("ignore")

def setup_logger():
    """Setup logger to always print time and level."""
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def model_based_metric(predicted_answer, example, model: BaseLLM, temperature: float = 0.01):
    if 'answers' in example:
        correct_answers = example['answers']
    else:
        raise ValueError(f"No answers found in example: {example}")

    prompt = f'We are assessing the quality of answers to the following question: {example["question"]}\n'
    if len(correct_answers) == 1:
        prompt += f"The expected answer is: {correct_answers[0]}.\n"
    else:
        prompt += f"The following are expected answers to this question: {correct_answers}.\n"

    prompt += f"The proposed answer is: {predicted_answer}\n"

    if len(correct_answers) == 1:
        prompt += "Within the context of the question, does the proposed answer mean the same as the expected answer?"
    else:
        prompt += "Within the context of the question, does the proposed answer mean the same as any of the expected answers?"

    prompt += " Respond only with yes or no.\nResponse:"

    judge_result = model.predict(prompt, temperature)
    if 'yes' in judge_result.lower():
        return 1.0
    elif 'no' in judge_result.lower():
        return 0.0
    else:
        logging.warning('Redo llm check.')
        # 1.0 表示 yes，0.0 表示 no
        judge_result = model.predict(prompt, 1) # 温度从 默认值（0.01） 提高到 1.0
        if 'yes' in judge_result.lower():
            return 1.0
        elif 'no' in judge_result.lower():
            return 0.0

        logging.warning('Answer neither no nor yes. Defaulting to no!')
        return 0.0


def get_parser():
    '''
    示例：
    python evaluate_responses.py --eval_model_type ollama --eval_model_name qwen2.5:72b-instruct-q4_0 --temperature 0.01 --scores_key qwen_scores --input_dir output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --output_dir output/eval/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --dataset squad --split train --num_samples 2000 --no-override
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_model_type", type=str, default="ollama")
    parser.add_argument("--eval_model_name", type=str, default="qwen2.5:72b-instruct-q4_0")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--scores_key", type=str, default="qwen_scores")
    parser.add_argument("--input_dir", type=str, default="output/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--output_dir", type=str, default="output/eval/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument('--dataset', type=str, default='squad')
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--num_samples', type=int, default=2000)
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
    return parser

def main(args):
    assert args.eval_model_type == "ollama", f"Only ollama model is supported for now. Got {args.eval_model_type}"
    eval_model = OllamaLLM(args.eval_model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    dataset = load_ds(args.dataset, args.split)
    data_dict = defaultdict(dict)
    for i in tqdm(range(args.num_samples)):
        data = dataset[i]
        data_dict[data['id']] = data
    
    for file in tqdm(os.listdir(args.input_dir), desc=f"Evaluating"):
        # 判断结果是否已存在
        if not args.override: # 若不覆盖
            output_path = os.path.join(args.output_dir, file)
            if os.path.exists(output_path): # 若结果已存在
                continue # 跳过

        input_path = os.path.join(args.input_dir, file)
        data = load_pickle_file(input_path)
        example_id = data['example_id']
        assert example_id in data_dict
        example = data_dict[example_id]

        scores = []
        for response in data['responses']:
            evaluate_score = model_based_metric(response.get('text', ''), example, eval_model, args.temperature)
            scores.append(evaluate_score)

        output_path = os.path.join(args.output_dir, file)
        if os.path.exists(output_path):
            result = load_pickle_file(output_path)    
        else:
            result = {
                'example_id': data['example_id'],
            }
        result[args.scores_key] = scores
        save_pickle_file(output_path, result)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    main(args)
    print('Done')
