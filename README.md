# FRAG: Federated Retrieval-augmented Generation

# latest update
- 2024-11-21: frag3重构开始

# 模型简称表
| 模型简称 | 模型全称 |
| -------- | -------- |
| - | Qwen/Qwen2.5-1.5B-Instruct |

# 数据集简称表
| 数据集简称 | 数据集全称 |
| -------- | -------- |
| squad | squad_v2 |

# 输出路径说明
- 所有输出都保存在`./output`目录下
- 每个模型一个子目录，如`./output/Qwen/Qwen2.5-1.5B-Instruct`
- 每个模型子目录下有数据集子目录`$dataset_name`，如`./output/Qwen/Qwen2.5-1.5B-Instruct/squad`
- 每个数据集子目录下有6个生成结果目录，分别是：
    - `greedy_golden`：贪婪带上下文（+隐状态）x1
    - `greedy_without`：贪婪不带上下文（+隐状态）x1
    - `greedy_irrelevant`：贪婪带无关上下文（+隐状态）x1
    - `sample_golden`：高温带上下文x10
    - `sample_without`：高温不带上下文x10
    - `sample_irrelevant`：高温带无关上下文x10
- 每个生成结果目录下有`$example_id.pkl`文件，保存生成结果
- 路径示例：`./output/Qwen/Qwen2.5-1.5B-Instruct/squad/greedy_golden/1.pkl`
- 每个`$example_id.pkl`文件是一个`dict`


# setup
```shell
conda-env update -f frag_enviroment.yaml
conda activate frag
```
# generate.py
- 生成答案

# output目录规则
- `output`表示根目录，默认存放`train`的结果
- `output/clustered`表示聚类结果，重写
- `output/eval`表示对生成答案的评判结果，追加（部分覆盖）
- `output/result`表示计算结果，追加（部分覆盖），TODO：全覆盖，这里应该放各种计算结果的合并结果，因此重新执行代价应该足够小