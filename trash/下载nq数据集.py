from abc import abstractmethod
import gc
import logging
import random
from tqdm import tqdm
from transformers import StoppingCriteria
import numpy as np
from collections import defaultdict
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteriaList

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

import argparse
import pickle

import logging
import os
import json
import hashlib
import datasets

from collections import Counter
import matplotlib.pyplot as plt

# 使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 加载数据集
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/natural_questions", "default")
print(dataset)

