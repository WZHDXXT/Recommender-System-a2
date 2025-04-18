import torch

# Recall@K
def recall_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]  
    relevant_items = set(relevant_items)  
    intersection = [item for item in recommended_items if item in relevant_items]
    return len(intersection) / k

def ndcg_at_k(recommended_items, relevant_items, k=10):
    recommended_items = recommended_items[:k]
    relevant_items = set(relevant_items)

    dcg = 0.0
    for i, item in enumerate(recommended_items):
        if item in relevant_items:
            dcg += 1 / torch.log2(torch.tensor(i + 2.0))  

    idcg = sum(1 / torch.log2(torch.tensor(i + 2.0)) for i in range(min(k, len(relevant_items))))
    return dcg / idcg if idcg > 0 else 0.0