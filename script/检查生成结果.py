import os
import pickle
import dotenv
dotenv.load_dotenv()

HOME_DIR = os.getenv("HOME_DIR", None)
assert HOME_DIR, "Please set HOME_DIR in .env file"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
DATASET_NAME = "squad"
SAMPLE_TYPE = "sample_without"
'''
    - `greedy_golden`：贪婪带上下文（+隐状态）x1
    - `greedy_without`：贪婪不带上下文（+隐状态）x1
    - `greedy_irrelevant`：贪婪带无关上下文（+隐状态）x1
    - `sample_golden`：高温带上下文x10
    - `sample_without`：高温不带上下文x10
    - `sample_irrelevant`：高温带无关上下文x10
'''

print(f"model:{MODEL_NAME}, dataset:{DATASET_NAME}, sample_type:{SAMPLE_TYPE}")

output_dir = f"{HOME_DIR}/code/FRAG/frag3/output/{MODEL_NAME}/{DATASET_NAME}/{SAMPLE_TYPE}"
for file in os.listdir(output_dir):
    if file.endswith(".pkl"):
        with open(os.path.join(output_dir, file), "rb") as f:
            result = pickle.load(f)
        responses = result['responses']
        for response in responses:
            # print('*', response["text"])
            print(responses)
            # break
    break

print("Done")
