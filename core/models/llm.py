"""Implement HuggingfaceModel models."""
import ast
import copy
import logging
import os
from collections import Counter

import torch

from transformers import AutoTokenizer
from transformers import AutoConfig
from transformers import AutoModelForCausalLM
from transformers import StoppingCriteria
from transformers import StoppingCriteriaList
from huggingface_hub import snapshot_download


from core.models.base_model import BaseModel

import ollama
import dotenv
dotenv.load_dotenv()

# 使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

class BaseLLM(BaseModel):
    def predict(self, input_data, temperature):
        raise NotImplementedError

    def get_p_true(self, input_data):
        # todo
        pass

class OllamaLLM(BaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name

    def predict(self, prompt: str, temperature: float = 0.01) -> str:
        response = ollama.generate(model=self.model_name, 
                                   prompt=prompt, 
                                   options={'temperature': temperature,}, 
                                   stream=False)
        return response['response']
    
