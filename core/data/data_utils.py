"""Data Loading Utilities."""
import logging
import os
import json
import hashlib
import datasets

import dotenv
dotenv.load_dotenv()

def load_ds(dataset_name, split):
    """Load dataset."""
    dataset = None
    if dataset_name in ["squad", "triviaqa"]:
        dataset_path = os.path.join(os.getenv("DATA_DIR"), f"song/{dataset_name}")
        dataset = datasets.load_from_disk(dataset_path)
    return dataset[split]
