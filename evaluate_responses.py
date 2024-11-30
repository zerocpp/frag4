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
from core.data.data_utils import load_ds, load_ds_from_json

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
    python evaluate_responses.py --eval_model_type ollama --eval_model_name qwen2.5:72b-instruct-q4_0 --temperature 0.01 --scores_key qwen_scores --input_dir output/train/generation/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --output_dir output/train/evaluation/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden --no-override --dataset_json_file data/squad_dev.json
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_model_type", type=str, default="ollama")
    parser.add_argument("--eval_model_name", type=str, default="qwen2.5:72b-instruct-q4_0")
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--scores_key", type=str, default="qwen_scores")
    parser.add_argument("--input_dir", type=str, default="output/train/generation/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--output_dir", type=str, default="output/train/evaluation/meta-llama/Llama-3.1-8B-Instruct/squad/sample_golden")
    parser.add_argument("--override", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset_json_file", type=str, default=None)
    return parser

def main(args):
    assert args.eval_model_type == "ollama", f"Only ollama model is supported for now. Got {args.eval_model_type}"
    eval_model = OllamaLLM(args.eval_model_name)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 加载数据集
    assert args.dataset_json_file, f"Dataset json file is required. Got {args.dataset_json_file}"
    id_list, data_dict = load_ds_from_json(args.dataset_json_file)
    
    for example_id in tqdm(id_list, desc="Evaluating"):
        assert example_id in data_dict, f"Example id {example_id} not found in dataset {args.dataset_json_file}."
        example = data_dict[example_id]
        eval_path = os.path.join(args.output_dir, f"{example_id}.pkl")
        if not args.override and os.path.exists(eval_path): # 若不覆盖且结果已存在
            continue

        gen_path = os.path.join(args.input_dir, f"{example_id}.pkl")
        if not os.path.exists(gen_path):
            # logging.warning(f"Generation result not found for example {example_id}. Skipping evaluation.")
            continue
        result = {
            'example_id': example_id,
            args.scores_key: [],
        }
        responses = load_pickle_file(gen_path)['responses']
        for response in responses:
            evaluate_score = model_based_metric(response.get('text', ''), example, eval_model, args.temperature)
            result[args.scores_key] = evaluate_score
        save_pickle_file(eval_path, result)

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    main(args)
    print('Done')
