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

def eval_beir_rerank_result(rank_result_path, entropy_result_path, dataset_path, dataset_name, k_values=[1,3,5,10]):
    # print(f"> Start evaluating rank results")
    # print(f"rank_result_path: {rank_result_path}")
    # print(f"dataset_path: {dataset_path}")
    # print(f"dataset_name: {dataset_name}")
    

    # Load data
    corpus, queries, qrels = load_data(dataset_path, dataset_name)

    def eval_scores(results):
        scores = {}
        #### Evaluate your retrieval using NDCG@k, MAP@K ...
        # print("Retriever evaluation")
        ndcg, _map, recall, precision = retriever.evaluate(qrels, results, k_values)
        # print('map:')
        # print(_map)
        scores['map'] = _map
        # print('precision:')
        # print(precision)
        scores['precision'] = precision
        # print('ndcg:')
        # print(ndcg)
        scores['ndcg'] = ndcg
        mrr = retriever.evaluate_custom(qrels, results, k_values, "mrr")
        # print('mrr:')
        # print(mrr)
        scores['mrr'] = mrr
        if dataset_name == "trec-covid":
            recall_cap = retriever.evaluate_custom(qrels, results, k_values, "recall_cap")
            # print('recall_cap:')
            # print(recall_cap)
            scores['recall_cap'] = recall_cap
        else:
            # print('recall:')
            # print(recall)
            scores['recall'] = recall
        return scores

    # rank_result_path = "/Users/song/Downloads/beir/nq/rank.tsv"
    rank_results = load_rank_results(rank_result_path)
    entropy_results = load_rank_results(entropy_result_path)
    rerank_results = rank_results.copy()
    suc1_count, suc_count, fail_count = 0, 0, 0
    for qid in rerank_results:
        for pid in rerank_results[qid]:
            if qid in entropy_results and pid in entropy_results[qid]:
                suc_count += 1
                if entropy_results[qid][pid] < 0.05:
                    rerank_results[qid][pid] += 1.0
                    suc1_count += 1
            else:
                fail_count += 1
    print(f"Success count: {suc_count}, success1 count: {suc1_count}, fail count: {fail_count}")

    retriever = EvaluateRetrieval()

    rank_scores = {}
    rank_scores['rank'] = eval_scores(rank_results)
    rank_scores['entropy'] = eval_scores(entropy_results)
    rank_scores['rerank'] = eval_scores(rerank_results)
    return rank_scores

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--rank_result_path', required=True)
    parser.add_argument('--entropy_result_path', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--dataset_name', required=True)
    args = parser.parse_args()
    eval_beir_rerank_result(args.rank_result_path, args.entropy_result_path, args.dataset_path, args.dataset_name)

