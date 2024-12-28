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
from rank_eval import eval_beir_rank_result

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

for SIZE_NAME in ["toy", "small"]:
    for dataset_name in tqdm(BEIR_DATASET_NAMES):
        try:
            all_scores[dataset_name] = {}
            print(f"> {dataset_name} rerank:")
            dataset_path = f'/home/song/dataset/beir/{dataset_name}'
            rerank_result_path = f'output/rerank/{dataset_name}/rerank-{SIZE_NAME}.tsv'
            print(f"rerank_result_path: {rerank_result_path}")
            rerank_scores = eval_beir_rank_result(rerank_result_path, dataset_path, dataset_name, k_values=[1,3,5,10])
            all_scores[dataset_name]["entropy"] = rerank_scores
            print(f">> {dataset_name} rank:")
            rank_result_path = f'/home/song/dataset/first/beir_rank/{dataset_name}/rank.tsv'
            rank_scores = eval_beir_rank_result(rank_result_path, dataset_path, dataset_name, k_values=[1,3,5,10])
            all_scores[dataset_name]["rank"] = rank_scores
        except Exception as e:
            print(f"Error: {e}")
    # Save all_scores
    save_pickle_file(f"output/rerank/all_scores_{SIZE_NAME}.pkl", all_scores)

print("ALL DONE!")

