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

# def strict_mutual_implication_check(entailment: tuple) -> bool:
#     assert len(entailment) == 2
#     assert all([entail in [0, 1, 2] for entail in entailment])
#     return entailment == (1, 1) # 两个都是蕴含才返回True

# def loose_mutual_implication_check(entailment: tuple) -> bool:
#     assert len(entailment) == 2
#     assert all([entail in [0, 1, 2] for entail in entailment])
#     return -1 not in entailment and entailment != (1, 1) # 两个都不是矛盾并且不能都是中立即返回True

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

# class EntailmentDeberta(BaseEntailment):
#     def __init__(self):
#         model_name = "cross-encoder/nli-deberta-v3-large"
#         self.model = CrossEncoder(model_name, 
#                                   max_length=2048,
#                                   device=DEVICE)

#     def check_implication(self, text1, text2, prefix=None) -> int:
#         '''
#         返回蕴含关系 0: contradiction, 1: neutral, 2: entailment
#         '''
#         if prefix:
#             text1 = f'{prefix} {text1}'
#             text2 = f'{prefix} {text2}'

#         logits = self.model.predict([(text1, text2)])
#         max_indexs = [max_index for max_index in logits.argmax(axis=1)]
#         # 统一index含义 ['contradiction', 'entailment', 'neutral'] -> ['contradiction', 'neutral', 'entailment'] 
#         index_mapping = {0: 0, 1: 2, 2: 1}
#         max_indexs = [index_mapping[max_index] for max_index in max_indexs]
#         return max_indexs[0]

#     def check_mutual_implication(self, text1, text2, prefix=None) -> tuple:
#         '''
#         返回双向蕴含关系，例如(2, 1) 其中 0: contradiction, 1: neutral, 2: entailment
#         '''
#         if prefix:
#             text1 = f'{prefix} {text1}'
#             text2 = f'{prefix} {text2}'

#         logits = self.model.predict([(text1, text2), (text2, text1)])
#         max_indexs = [max_index for max_index in logits.argmax(axis=1)]
#         # 统一index含义 ['contradiction', 'entailment', 'neutral'] -> ['contradiction', 'neutral', 'entailment'] 
#         index_mapping = {0: 0, 1: 2, 2: 1}
#         max_indexs = tuple(index_mapping[max_index] for max_index in max_indexs)
#         return max_indexs

# class EntailmentLLM(BaseEntailment):
#     def __init__(self, llm, entailment_file):
#         self.llm = llm
#         self.entailment_file = entailment_file
#         self.prediction_cache = defaultdict(dict)
#         if os.path.exists(self.entailment_file):
#             with open(self.entailment_file, 'rb') as f:
#                 self.prediction_cache = pickle.load(f)

#     def save_prediction_cache(self):
#         with open(self.entailment_file, 'wb') as f:
#             pickle.dump(self.prediction_cache, f)

#     @abstractmethod
#     def equivalence_prompt(self, text1, text2, question):
#         pass

#     def check_implication(self, text1, text2, example=None):
#         if example is None:
#             raise ValueError
#         prompt = self.equivalence_prompt(text1, text2, example['question'])
#         logging.info(f'Entailment prompt: {prompt}')
#         hashed = md5hash(prompt)
#         if hashed in self.prediction_cache:
#             response = self.prediction_cache[hashed]
#             logging.info(f'Entailment response from cache: {response}')
#         else:
#             response = self.llm.predict(prompt, temperature=0.02)
#             self.prediction_cache[hashed] = response
#             logging.info(f'Entailment response: {response}')

#         binary_response = response.lower()[:30]
#         if 'entailment' in binary_response:
#             return 2
#         elif 'neutral' in binary_response:
#             return 1
#         elif 'contradiction' in binary_response:
#             return 0
#         else:
#             logging.warning('MANUAL NEUTRAL!')
#             return 1

# class EntailmentOllama(EntailmentLLM):
#     def __init__(self, model_name):
#         self.llm = get_llm(model_name)
#         self.entailment_file = f'output/entailment_cache_{model_name}.pkl'
#         super().__init__(self.llm, self.entailment_file)

#     def equivalence_prompt(self, text1, text2, question):
#         # Llama 的 prompt 格式
#         prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
#         prompt += "Here are two possible answers:\n"
#         prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
#         prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
#         prompt += "Response:""" # GPT 的 prompt 不需要这句

#         return prompt

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
        # label_mapping = ['contradiction', 'neutral', 'entailment']
        # print(text1, '|', text2, '| check_implication', res1, label_mapping[res1])
        # res2 = entail.check_mutual_implication(text1, text2)
        # print(text1, '|', text2, '| mutual_implication', res2, [label_mapping[res] for res in res2])

    # from transformers import AutoTokenizer, AutoModelForSequenceClassification
    # import torch

    # model = AutoModelForSequenceClassification.from_pretrained('cross-encoder/nli-deberta-v3-large')
    # tokenizer = AutoTokenizer.from_pretrained('cross-encoder/nli-deberta-v3-large')

    # features = tokenizer(['A man is eating pizza', 'A black race car starts up in front of a crowd of people.'], ['A man eats something', 'A man is driving down a lonely road.'],  padding=True, truncation=True, return_tensors="pt")

    # model.eval()
    # with torch.no_grad():
    #     scores = model(**features).logits
    #     label_mapping = ['contradiction', 'entailment', 'neutral']
    #     labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
    #     print(labels)

