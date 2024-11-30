import os
import pickle
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()
import numpy as np
from collections import defaultdict
from core.data.data_utils import load_ds

datasets=["squad", "triviaqa"]
splits=["train", "test", "validation"]
num_samples = [10000, 1000, 1000]

for dataset in datasets:
    for split, num_sample in zip(splits, num_samples):
        print(f"{dataset} {split} {num_sample}")
        all_data = load_ds(dataset, split)
        new_data = {
            "id": [],
            "data": {},
        }
        for i in tqdm(range(num_sample)):
            data = all_data[i]
            new_data["id"].append(data["id"])
            new_data["data"][data["id"]] = data
        
        file = f"output/dataset/{dataset}_{split}.json"
        json.dump(new_data, open(file, "w"), indent=4)
