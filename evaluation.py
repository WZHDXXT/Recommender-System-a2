import numpy as np
import torch
import math
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


def evaluate(model, user_train, user_valid, max_len, make_sequence_dataset, bert4rec_dataset, K):
    model.eval()

    NDCG = 0.0
    RECALL = 0.0

    num_item_sample = 100
    users = [user for user in range(make_sequence_dataset.num_user)] 

    for user in users:
        ndcg_u = 0.0
        recall_u = 0.0
        count = 0

        seq_prefix = user_train[user]
        for target_item in user_valid[user]:
            seq_input = (seq_prefix + [make_sequence_dataset.num_item + 1])[-max_len:]
            padding_len = max_len - len(seq_input)
            seq_input = [0] * padding_len + seq_input

            rated = user_train[user] + user_valid[user]
            items = [target_item] + bert4rec_dataset.random_neg_sampling(rated_items=rated, num_samples=num_item_sample)

            with torch.no_grad():
                seq_tensor = torch.LongTensor([seq_input]).to(device)
                predictions = -model(seq_tensor)
                predictions = predictions[0][-1][items]
                rank = predictions.argsort().argsort()[0].item()

            if rank < K:
                ndcg_u += 1 / np.log2(rank + 2)
                recall_u += 1

            seq_prefix.append(target_item)
            count += 1

        NDCG += ndcg_u / count if count > 0 else 0
        RECALL += recall_u / count if count > 0 else 0

    total = len(users)
    return NDCG / total, RECALL / total