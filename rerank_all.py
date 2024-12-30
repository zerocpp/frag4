'''
对beir数据集所有子集进行rerank
'''
from rerank import rerank_by_entropy


BEIR_DATASET_NAMES = ["trec-covid", "climate-fever", "dbpedia-entity", "fever", "fiqa", "hotpotqa", "msmarco",  "nfcorpus", "nq", "scidocs", "scifact"]
SIZE_NAME = "small"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

for dataset_name in BEIR_DATASET_NAMES:
    dataset_path = f'dataset/rank/{dataset_name}/{dataset_name}-{SIZE_NAME}.jsonl'
    input_dir = f'output/rank/cluster/{MODEL_NAME}/{dataset_name}'
    output_path = f'output/rerank/{dataset_name}/entropy-{SIZE_NAME}.tsv'
    rerank_by_entropy(dataset_path, input_dir, output_path)
print("ALL DONE!")
