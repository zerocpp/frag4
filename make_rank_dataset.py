'''
构造rank数据集
'''
import os
import json
import jsonlines
import pandas as pd
from collections import defaultdict
import shutil
from tqdm import tqdm
from argparse import ArgumentParser

def load_dataset(dataset_path):
    query_path = os.path.join(dataset_path, 'queries.jsonl')
    queries = {}
    with jsonlines.open(query_path) as reader:
        for query in reader:
            queries[query['_id']] = query['text']

    doc_path = os.path.join(dataset_path, 'corpus.jsonl')
    docs = {}
    with jsonlines.open(doc_path) as reader:
        for doc in reader:
            docs[doc['_id']] = doc['text']

    rel_path = os.path.join(dataset_path, 'qrels/test.tsv')
    df = pd.read_csv(rel_path, sep='\t', header=0)
    
    scores = defaultdict(dict)
    for qid, docid, score in df.values:
        scores[qid][docid] = score
    
    return queries, docs, scores

def load_rank_results(rank_result_path, rank_num):
    df = pd.read_csv(rank_result_path, sep='\t', header=None, names=['query-id', 'corpus-id', 'score'])
    rank_results = defaultdict(list)
    for qid, docid, score in df.values:
        if len(rank_results[qid]) < rank_num:
            rank_results[qid].append(docid)
    return rank_results

def make_rank_dataset(dataset_name, dataset_path, rank_result_path, output_path, rank_num):
    queries, docs, scores = load_dataset(dataset_path)
    rank_results = load_rank_results(rank_result_path, rank_num)

    qids = sorted(list(rank_results.keys()))
    with jsonlines.open(output_path, 'w') as writer:
        for qid in tqdm(qids):
            query = queries[qid]
            doc_ids = ['no'] + rank_results[qid]
            for doc_id in doc_ids:
                assert doc_id in docs or doc_id == 'no'
                writer.write({
                    'metadata': {
                        'dataset': dataset_name,
                        'query_id': qid,
                        'doc_id': doc_id,
                    },
                    'id': f'nq-{qid}-{doc_id}',
                    'question': query,
                    'context': docs.get(doc_id, None),
                })

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--rank_result_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--rank_num', type=int, default=100)
    args = parser.parse_args()

    make_rank_dataset(args.dataset_name, args.dataset_path, args.rank_result_path, args.output_path, args.rank_num)
    print(f"Rank dataset saved to {args.output_path}")
    # python make_rank_dataset.py --dataset_name nq --dataset_path /Users/song/Downloads/beir/nq --rank_result_path /Users/song/Downloads/beir/nq/rank.tsv --rank_num 10 --output_path /Users/song/Downloads/beir/nq/rank_dataset.jsonl
