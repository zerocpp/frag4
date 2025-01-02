import gc
import os
import pickle
import jsonlines
import torch
from tqdm import tqdm
import pandas as pd
import csv
from collections import defaultdict
import argparse
from core.models.entailment import EntailmentDeberta
from rank_eval import load_rank_results
import os
import pickle
from core.computation.uncertainty_measure import cluster_assignment_entropy

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

def merge_score(rank_score, entropy_score, no_doc_entropy):
    if entropy_score is None:
        return rank_score
    if no_doc_entropy is None:
        return rank_score
    if entropy_score >= no_doc_entropy:
        return rank_score
    if entropy_score < 0.01:
        return rank_score + 1.0
    return rank_score

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
    
    scores = defaultdict(dict)
    for qid, docid, score in df.values:
        scores[str(qid)][str(docid)] = score
    
    return queries, docs, scores

size_name = "large"
dataset_names = ["trec-covid", "climate-fever", "dbpedia-entity", "fever", "hotpotqa", "nfcorpus", "nq", "scidocs"]
for dataset_name in tqdm(dataset_names, desc='dataset'):
    dataset_path = f'/home/song/dataset/beir/{dataset_name}'
    queries, docs, scores = load_dataset(dataset_path)
    rank_result_path = f'dataset/rank/{dataset_name}/{dataset_name}-rank10-{size_name}.tsv'
    rank_results = load_rank_results(rank_result_path)
    
    def get_no_doc(qid):
        docid = 'no'
        samples = load_samples(dataset_name, qid, docid)
        samples_text = '|'.join(samples)
        cluster_ids = load_cluster_ids(dataset_name, qid, docid)
        cluster_text = '|'.join([str(x) for x in cluster_ids])
        entropy = compute_entropy(cluster_ids)
        return (samples_text, cluster_text, entropy)

    merge_results = []
    for qid in rank_results:
        no_doc_samples, no_doc_clusters, no_doc_entropy = get_no_doc(qid)
        items = []
        for i, docid in enumerate(rank_results[qid]):
            samples = load_samples(dataset_name, qid, docid)
            samples_text = '|'.join(samples)
            cluster_ids = load_cluster_ids(dataset_name, qid, docid)
            cluster_text = '|'.join([str(x) for x in cluster_ids])
            entropy = compute_entropy(cluster_ids)
            items.append([str(qid), # qid
                            queries.get(str(qid), ''), # query
                            str(docid), # docid
                            docs.get(str(docid), ''), # doc
                            scores.get(str(qid), {}).get(str(docid), 0.0), # gold_score
                            rank_results.get(qid, {}).get(docid, 0.0), # rank_score
                            entropy, # doc_entropy
                            no_doc_entropy, # no_doc_entropy
                            merge_score(rank_results.get(qid, {}).get(docid, 0.0), entropy, no_doc_entropy), # merge_score
                            samples_text, # doc_samples
                            cluster_text, # doc_clusters
                            no_doc_samples, # no_doc_samples
                            no_doc_clusters, # no_doc_clusters
                            i, # rank_index
                            i, # merge_index
                            0, # diff_index
                            ])
        # 按照merge_score排序
        items.sort(key=lambda x: x[8], reverse=True)
        for i, item in enumerate(items):
            item[-2] = i
            item[-1] = i - item[-3]
        merge_results.extend(items)
    output_file_path = f'output/export/{dataset_name}-{size_name}.tsv'
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['qid', 'query', 'docid', 'doc', 'gold_score', 'rank_score', 'doc_entropy', 'no_doc_entropy', 'merge_score', 'doc_samples', 'doc_clusters', 'no_doc_samples', 'no_doc_clusters', 'rank_index', 'merge_index', 'diff_index'])
        writer.writerows(merge_results)
    print(output_file_path)
