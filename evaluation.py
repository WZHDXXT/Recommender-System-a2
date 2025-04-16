import torch

# Recall@K
def recall_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k].flatten().tolist()
    relevant_items = relevant_items.flatten().tolist()
    recommended_items = set(recommended_items)
    relevant_items = set(relevant_items)  
    intersection = recommended_items.intersection(relevant_items)
    return len(intersection) / k

# NDCG@K
def ndcg_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k].flatten().tolist()
    relevant_items = relevant_items.flatten().tolist()
    recommended_items = set(recommended_items)
    relevant_items = set(relevant_items)  
    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1 / torch.log2(torch.tensor(i + 2, dtype=torch.float32))
    idcg = 0.0
    for i in range(min(k, len(relevant_items))):
        idcg += 1 / torch.log2(torch.tensor(i + 2, dtype=torch.float32))
    return dcg / idcg if idcg > 0 else 0.0
