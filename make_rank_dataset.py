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

def load_rank_results(rank_result_path):
    df = pd.read_csv(rank_result_path, sep='\t', header=None, names=['query-id', 'corpus-id', 'score'])
    rank_results = defaultdict(list)
    for qid, docid, score in df.values:
        rank_results[str(qid)].append(str(docid))
    return rank_results

def make_rank_dataset(dataset_name, dataset_path, rank_result_path, output_path, query_num, rank_num):
    print(f"######## Dataset {dataset_name}")

    queries, docs, scores = load_dataset(dataset_path)
    rank_results = load_rank_results(rank_result_path)
    # print(type(rank_results), len(rank_results), rank_results[list(rank_results.keys())[0]])

    rel_qids = set(map(str, scores.keys())) # 有标注的问题
    rank_qids = set(map(str, rank_results.keys())) # 有rank结果的问题
    qids = rel_qids & rank_qids # 有标注且有rank结果的问题
    print(f"Total {len(rel_qids)} questions have relevance results")
    print(f"Total {len(rank_qids)} questions have rank results")
    print(f"Total {len(qids)} questions have both relevance and rank results")

    qids = sorted(list(qids))[:query_num] # 只取前query_num个问题
    print(f"> Use {len(qids)} questions")
    prompt_count = 0 # 实际生成的prompt行数
    with jsonlines.open(output_path, 'w') as writer:
        for qid in tqdm(qids):
            query = queries[qid]
            doc_ids = ['no'] + rank_results[qid][:rank_num] # no:不使用文档, rank_num:只使用前rank_num个文档
            # print(f"len(rank_results[qid]): {len(rank_results[qid])}")
            # print(f"doc_ids: {len(doc_ids)}")
            for doc_id in doc_ids:
                assert doc_id in docs or doc_id == 'no'
                writer.write({
                    'metadata': {
                        'dataset': dataset_name,
                        'query_id': qid,
                        'doc_id': doc_id,
                    },
                    'id': f'{dataset_name}-{qid}-{doc_id}',
                    'question': query,
                    'context': docs.get(doc_id, None),
                })
                prompt_count += 1
    print(f"Saved to {output_path}")
    print(f"Total {prompt_count} prompts")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset_name', required=True)
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--rank_result_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--query_num', type=int, required=True)
    parser.add_argument('--rank_num', type=int, required=True)
    args = parser.parse_args()

    make_rank_dataset(args.dataset_name, args.dataset_path, args.rank_result_path, args.output_path, args.query_num, args.rank_num)
    print("Done!")

# !python make_rank_dataset.py --dataset_path $dataset_path --rank_result_path $rank_result_path --output_path $output_path --query_num $query_num --rank_num $rank_num