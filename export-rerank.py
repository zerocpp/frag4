import os
import pickle
import jsonlines
from tqdm import tqdm
from time import time
import pandas as pd
import csv
from collections import defaultdict
from rank_eval import load_rank_results, eval_rank_results
import os
import pickle
from core.computation.uncertainty_measure import cluster_assignment_entropy


# ALL_DATASET_NAMES = ["trec-covid", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "msmarco", "nfcorpus", "scidocs", "scifact", "nq"]
# SIZE_NAME = "500"
ALL_DATASET_NAMES = ["nq"]
SIZE_NAME = "all"
# SIZE_NAME = "large"
BEIR_DATASET_DIR = "/home/song/dataset/beir"
RANK_DIR = "dataset/rank"
SAMPLE_DIR = "output/sample"

def load_pickle_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pickle_file(file_path, data):
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)

def load_samples(dataset_name, qid, doc_id):
    file_path = f'output/rank/gen/Qwen/Qwen2.5-7B-Instruct/{dataset_name}/{dataset_name}-{qid}-{doc_id}.pkl'
    if os.path.exists(file_path):
        result = load_pickle_file(file_path)
        return [x['text'] for x in result['sample']]
    return []

def load_greedy(dataset_name, qid, doc_id):
    file_path = f'output/rank/gen/Qwen/Qwen2.5-7B-Instruct/{dataset_name}/{dataset_name}-{qid}-{doc_id}.pkl'
    if os.path.exists(file_path):
        result = load_pickle_file(file_path)
        return result.get('greedy', {}).get('text', None)
    return None


def load_cluster_ids(dataset_name, qid, doc_id):
    file_path = f'output/rank/cluster/Qwen/Qwen2.5-7B-Instruct/{dataset_name}/{dataset_name}-{qid}-{doc_id}.pkl'
    if os.path.exists(file_path):
        result = load_pickle_file(file_path)
        return result.get('cluster_ids', [])
    return []


# 计算语义熵
def compute_entropy(cluster_ids):
    if len(cluster_ids) == 0:
        return None
    return cluster_assignment_entropy(cluster_ids)


def load_dataset(dataset_path):
    query_path = os.path.join(dataset_path, 'queries.jsonl')
    queries = {}
    with jsonlines.open(query_path) as reader:
        for query in reader:
            queries[str(query['_id'])] = query['text']

    doc_path = os.path.join(dataset_path, 'corpus.jsonl')
    docs = {}
    with jsonlines.open(doc_path) as reader:
        for doc in reader:
            docs[str(doc['_id'])] = doc['text']

    rel_path = os.path.join(dataset_path, 'qrels/test.tsv')
    df = pd.read_csv(rel_path, sep='\t', header=0)
    
    qrels = defaultdict(dict)
    for qid, docid, score in df.values:
        qrels[str(qid)][str(docid)] = score
    
    return queries, docs, dict(qrels)

def make_sample_data(rank_results, dataset_name):
    sample_data = defaultdict(dict)
    for qid, doc_ids in tqdm(rank_results.items()):
        for doc_id in ['no']+list(doc_ids.keys()):
            samples = load_samples(dataset_name, qid, doc_id)
            greedy = load_greedy(dataset_name, qid, doc_id)
            cluster_ids = load_cluster_ids(dataset_name, qid, doc_id)
            sample_data[qid][doc_id] = {
                'greedy': greedy,
                'samples': samples,
                'cluster_ids': cluster_ids,
                'entropy': compute_entropy(cluster_ids)
            }
    return dict(sample_data)

def make_rank_results(dataset_name):
    rank_path = f'{RANK_DIR}/{dataset_name}/{dataset_name}-rank10-{SIZE_NAME}.tsv'
    rank_results = load_rank_results(rank_path)
    return rank_results

def main():
    for dataset_name in ALL_DATASET_NAMES:
        try:
            sample_path = f'{SAMPLE_DIR}/{dataset_name}/sample.pkl'
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)

            rank_results = make_rank_results(dataset_name)
            sample_data = make_sample_data(rank_results, dataset_name)
            save_pickle_file(sample_path, sample_data)
            print(f'{dataset_name} sample data saved to {sample_path}')
        except Exception as e:
            print(f'{dataset_name} error: {e}')

if __name__ == '__main__':
    start_time = time()
    main()
    print(f'elapsed time: {time()-start_time:.2f} seconds')
    print('done')
