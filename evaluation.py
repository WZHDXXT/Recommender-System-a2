import numpy as np
from tqdm import tqdm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_(model, user_train, user_valid, max_len, make_sequence_dataset, bert4rec_dataset):
    model.eval()

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0

    num_item_sample = 100
    users = [user for user in range(make_sequence_dataset.num_user)]

    for user in users:
        # seq = (user_train[user] + [make_sequence_dataset.num_item + 1])[-max_len:]
        seq = user_valid[user][-max_len:]
        padding_len = max_len - len(seq)
        seq = [0] * padding_len + seq

        # rated = user_train[user] + [user_valid[user]]
        rated = user_valid[user]
        items = [user_valid[user][-1]] + bert4rec_dataset.random_neg_sampling(rated_items=rated, num_samples=num_item_sample)

        with torch.no_grad():
            seq = torch.LongTensor([seq]).to(device)
            predictions = -model(seq)
            predictions = predictions[0][-1][items]
            rank = predictions.argsort().argsort()[0].item()
            

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HIT += 1
            RECALL += 1 / 1  # only 1 relevant item → recall = 1 if hit

    total = len(users)
    return NDCG / total, HIT / total, RECALL / total


def evaluate(model, user_train, user_valid, max_len, make_sequence_dataset, bert4rec_dataset):
    model.eval()

    NDCG = 0.0
    HIT = 0.0
    RECALL = 0.0

    num_item_sample = 100
    users = [user for user in range(make_sequence_dataset.num_user)] 

    for user in users:
        seq = user_valid[user][:-1][-max_len:] 
        padding_len = max_len - len(seq)
        seq = [0] * padding_len + seq

        relevant_items = [user_valid[user][-1]]

        rated = user_train[user] + user_valid[user]
        
        items = relevant_items + bert4rec_dataset.random_neg_sampling(rated_items=rated, num_samples=num_item_sample)
        # items = relevant_items
        with torch.no_grad():
            seq = torch.LongTensor([seq]).to(device)
            predictions = -model(seq)
            # predicted score at the last position
            predictions = predictions[0][-1][items]
            top_k_indices = predictions.argsort(descending=True)[:10].tolist()
            top_k_preds = [items[i] for i in top_k_indices]

        #  Recall、NDCG, HIT
        hit_count = len(set(relevant_items) & set(top_k_preds))  
        if hit_count > 0:
            HIT += 1 
            RECALL += hit_count / len(relevant_items) 
            NDCG += sum([1 / np.log2(i + 2) for i, item in enumerate(top_k_preds) if item in relevant_items]) 

    total = len(users)
    return NDCG / total, HIT / total, RECALL / total