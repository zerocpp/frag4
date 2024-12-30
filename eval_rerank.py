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
from rank_eval import eval_beir_rerank_result

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

BEIR_DATASET_NAMES = ["trec-covid", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "msmarco",  "nfcorpus", "nq", "scidocs", "scifact"]
# SIZE_NAME = "toy"
# SIZE_NAME = "small"

all_scores = defaultdict(dict)

for SIZE_NAME in ["small"]:
    for dataset_name in tqdm(BEIR_DATASET_NAMES):
        try:
            dataset_path = f'/home/song/dataset/beir/{dataset_name}'
            rank_result_path = f'dataset/rank/{dataset_name}/{dataset_name}-rank10.tsv'
            entropy_result_path = f'output/rerank/{dataset_name}/entropy-{SIZE_NAME}.tsv'
            all_scores[dataset_name] = eval_beir_rerank_result(rank_result_path, entropy_result_path, dataset_path, dataset_name, k_values=[1,3,5,10])
        except Exception as e:
            print(f"Error: {e}")
    # Save all_scores
    save_pickle_file(f"output/rerank/entropy_scores_{SIZE_NAME}.pkl", all_scores)

print("ALL DONE!")

