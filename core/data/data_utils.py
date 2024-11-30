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

def load_ds_from_json(file_path):
    """
    Load dataset from json file.
    {
        "id": ["572f5533a23a5019007fc55d", ...],
        "data": {
            "57313c0a497a881900248cae": {
                "id": "57313c0a497a881900248cae",
                "question": "How were pigments like red lake created?",
                "context": "Unlike vermilion or red ochre, made from minerals, red lake pigments are made by mixing organic dyes, made from insects or plants, with white chalk or alum. Red lac was made from the gum lac, the dark red resinous substance secreted by various scale insects, particularly the Laccifer lacca from India. Carmine lake was made from the cochineal insect from Central and South America, Kermes lake came from a different scale insect, kermes vermilio, which thrived on oak trees around the Mediterranean. Other red lakes were made from the rose madder plant and from the brazilwood tree.",
                "answers": [
                    "mixing organic dyes"
                ],
                "irrelevant_id": "57268318708984140094c88b",
                "irrelevant_context": "One of the paper's best known front pages, published on 4 May 1982, commemorated the torpedoing of the Argentine ship the General Belgrano by running the story under the headline \"GOTCHA\". At MacKenzie's insistence, and against the wishes of Murdoch (the mogul was present because almost all the journalists were on strike), the headline was changed for later editions after the extent of Argentinian casualties became known. John Shirley, a reporter for The Sunday Times, witnessed copies of this edition of The Sun being thrown overboard by sailors and marines on HMS Fearless."
            }, ...
        }
    }
    """
    with open(file_path, 'r') as f:
        dataset = json.load(f)
    return dataset["id"], dataset["data"]
