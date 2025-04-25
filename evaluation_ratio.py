import numpy as np
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
                top_scores = model(seq_tensor)[0, -1][items]
                rank = (top_scores).argsort(descending=True).tolist().index(0)
            if rank < K:
                ndcg_u += 1 / np.log2(rank + 2)
                recall_u += 1

            seq_prefix.append(target_item)
            count += 1

        NDCG += ndcg_u / count if count > 0 else 0
        RECALL += recall_u / count if count > 0 else 0

    total = len(users)
    return NDCG / total, RECALL / total


def full_ranking_evaluate_with_validation(model, user_train, user_valid, user_test, max_len, vocab_size, device, K=10):
    model.eval()
    recall_sum, ndcg_sum, user_count = 0.0, 0.0, 0

    for user in user_test:
        if len(user_train[user]) < 1 or len(user_test[user]) < 1:
            continue

        seq = user_train[user] + user_valid[user]  # train + valid 

        for target_item in user_test[user]:
            input_seq = (seq + [vocab_size - 1])[-max_len:]  
            input_seq = [0] * (max_len - len(input_seq)) + input_seq

            seq_tensor = torch.LongTensor([input_seq]).to(device)

            with torch.no_grad():
                logits = model(seq_tensor)  # [1, seq_len, vocab_size]
                pred_scores = logits[0, -1] 

            topk = torch.topk(pred_scores, k=K).indices.cpu().tolist()

            if target_item in topk:
                recall_sum += 1
                rank = topk.index(target_item)
                ndcg_sum += 1 / torch.log2(torch.tensor(rank + 2.0)).item()

            seq.append(target_item)

        user_count += 1

    return ndcg_sum / user_count, recall_sum / user_count

def evaluate_test(model, user_train, user_valid, user_test, max_len, make_sequence_dataset, bert4rec_dataset, K):
    model.eval()

    NDCG = 0.0
    RECALL = 0.0

    num_item_sample = 100
    users = [user for user in range(make_sequence_dataset.num_user)] 

    for user in users:
        ndcg_u = 0.0
        recall_u = 0.0
        count = 0

        seq_prefix = user_train[user] + user_valid[user]
        for target_item in user_test[user]:
            seq_input = (seq_prefix + [make_sequence_dataset.num_item + 1])[-max_len:]
            padding_len = max_len - len(seq_input)
            seq_input = [0] * padding_len + seq_input

            rated = user_train[user] + user_valid[user] + user_test[user]
            items = [target_item] + bert4rec_dataset.random_neg_sampling(rated_items=rated, num_samples=num_item_sample)

            with torch.no_grad():
                seq_tensor = torch.LongTensor([seq_input]).to(device)
                top_scores = model(seq_tensor)[0, -1][items]
                rank = (top_scores).argsort(descending=True).tolist().index(0)

            if rank < K:
                ndcg_u += 1 / np.log2(rank + 2)
                recall_u += 1

            seq_prefix.append(target_item)
            count += 1

        NDCG += ndcg_u / count if count > 0 else 0
        RECALL += recall_u / count if count > 0 else 0

    total = len(users)
    return NDCG / total, RECALL / total