from beir.reranking import Rerank
from beir.reranking.models import CrossEncoder
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
import csv
import os
import logging
import json
from argparse import ArgumentParser

def load_rank_results(rank_result_path):
    csv_reader = csv.reader(open(rank_result_path), delimiter="\t", quotechar='|')
    results = dict()
    for row in csv_reader:
        qid = str(row[0])
        pid = str(row[1])
        score = float(row[2])
        if qid not in results:
            results[qid] = dict()
        results[qid][pid] = score
    return results

def load_data(dataset_path, dataset_name):
    # if dataset_name == "msmarco": # why?
    #     corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="dev")
    # else:
    #     corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
    corpus, queries, qrels = GenericDataLoader(data_folder=dataset_path).load(split="test")
    return corpus, queries, qrels

def eval_beir_rank_result(rank_result_path, dataset_path, dataset_name):
    # Load data
    corpus, queries, qrels = load_data(dataset_path, dataset_name)

    # rank_result_path = "/Users/song/Downloads/beir/nq/rank.tsv"
    results = load_rank_results(rank_result_path)
    
    retriever = EvaluateRetrieval()
        
    #### Evaluate your retrieval using NDCG@k, MAP@K ...
    print("Retriever evaluation")
    ndcg, _map, recall, precision = retriever.evaluate(qrels, results, [1,3,5,10,20,100])
    print('map:')
    print(_map)
    print('precision:')
    print(precision)
    print('ndcg:')
    print(ndcg)
    mrr = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100], "mrr")
    print('mrr:')
    print(mrr)
    if dataset_name == "trec-covid":
        recall_cap = retriever.evaluate_custom(qrels, results, [1,3,5,10,20,100], "recall_cap")
        print('recall_cap:')
        print(recall_cap)
    else:
        print('recall:')
        print(recall)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--rank_result_path', required=True)
    args = parser.parse_args()
    eval_beir_rank_result(args.rank_result_path, args.dataset_path, args.dataset_name)
    # python rank_eval.py --dataset_name nq --dataset_path '/Users/song/Downloads/nq' --rank_result_path '/Users/song/Downloads/beir/nq/rank.tsv'
