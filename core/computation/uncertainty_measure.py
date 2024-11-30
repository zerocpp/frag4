'''
计算不确定性
'''
import numpy as np

def cluster_assignment_entropy(semantic_ids):
    """根据不同簇被分配的频率来估计语义不确定性。

    我们从语义ID估计簇分配的类别分布。不确定性由该分布的熵给出。
    这个估计不使用词元似然，而是完全依赖于簇分配。如果概率质量分散在
    多个簇之间，熵就会更大。如果概率质量集中在少数几个簇上，熵就会更小。

    输入:
        semantic_ids: 语义ID列表,例如 [0, 1, 2, 1]。
    输出:
        cluster_entropy: 熵值,例如对于 p = [1/4, 2/4, 1/4] 计算 (-p log p).sum()。
        若所有样本都分配到同一个簇，则熵为0（若样本数为0，则熵也为0）。
        若样本均匀分布在所有簇上，则熵最大。
    """
    if len(semantic_ids) == 0:
        return 0
    
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy
