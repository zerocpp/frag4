from abc import abstractmethod
import ast
import hashlib
import logging
import os
import pickle
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sentence_transformers import CrossEncoder
import torch
from collections import defaultdict
import torch.nn.functional as F

# 使用镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 默认优先级: cuda > mps > cpu
DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
# 设备
DEVICE = os.getenv("DEVICE", DEFAULT_DEVICE)

def md5hash(s: str) -> str:
    '''将字符串s序列化'''
    return str(int(hashlib.md5(str(s).encode('utf-8')).hexdigest(), 16))

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self, model_name="microsoft/deberta-v2-xxlarge-mnli"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name).to(DEVICE)

    def check_implication(self, text1, text2, prefix=None) -> int:
        '''
        ("A man is eating a sandwich.", "A man is eating food."),  # 2 (Entailment)
        ("A man is eating a sandwich.", "A woman is eating a sandwich."),  # 0 (Contradiction)
        ("A man is eating a sandwich.", "A man is sitting on a bench."),  # 1 (Neutral)
        '''
        if prefix:
            text1 = f'{prefix} {text1}'
            text2 = f'{prefix} {text2}'
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        # if os.environ.get('DEBERTA_FULL_LOG', False):
        #     logging.info('Deberta Input: %s -> %s', text1, text2)
        #     logging.info('Deberta Prediction: %s', prediction)

        return prediction


if __name__ == '__main__':
    entail = EntailmentDeberta()
    print('entail model loaded')

        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2

    for text1, text2 in [('I am a student', 'I am a teacher'),
        ('I am a boy', 'I am a girl'),
        ('Paris is capital of France', 'Beijing is capital of France'),
        ('A man is eating pizza', 'A black race car starts up in front of a crowd of people.'),
        ('A man eats something', 'A man is driving down a lonely road.'),
        ('I am a boy', 'I am a boy'),
        ('Good', 'Great'),
        ('How many people are there in the world? 70 billion', 'How many people are there in the world? about 70 billion'),
        ('The weather is good', 'The weather is good and I like you'), # 1
        ('The weather is good and I like you', 'The weather is good'), # 2
        ("A man is eating a sandwich.", "A man is eating food."),  # 2 (Entailment)
("A man is eating a sandwich.", "A woman is eating a sandwich."),  # 0 (Contradiction)
("A man is eating a sandwich.", "A man is sitting on a bench."),  # 1 (Neutral)
        ]:
        res1 = entail.check_implication(text1, text2)
        print(text1, '|', text2, '| check_implication', res1)
