from bert4rec import BERT4Rec  
from maskdataset import BERT4RecDataset, collate_fn, EvalDataset
from torch.utils.data import DataLoader
from data_load_ratio import data_load
import torch.nn as nn
import torch
import torch.optim as optim
from evaluation import recall_at_k, ndcg_at_k

user_train, user_valid, user_test, usernum, itemnum = data_load('ratings.dat')

epochs = 10
max_len = 20
vocab_size = itemnum + 2  # +1 for padding, +1 for [MASK]
mask_token = vocab_size - 1  # index of [MASK], no conflicting with existing item ids

model = BERT4Rec(vocab_size=vocab_size, max_seq_length=max_len)
model.train()

train_dataset = BERT4RecDataset(user_train, max_len=max_len, mask_token=mask_token)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)


for epoch in range(epochs):
    model.train()
    for input_ids, masked_pos, masked_labels in train_loader:
        logits = model(input_ids, masked_pos)
        loss = criterion(logits.view(-1, vocab_size), masked_labels.view(-1).long())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# only mask's last position
eval_dataset = EvalDataset(user_valid, max_len=max_len, mask_token=mask_token)
eval_loader = DataLoader(eval_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# evaluation
model.eval() 

recall_sum = 0
ndcg_sum = 0
num_samples = 0

for input_ids, masked_pos, masked_labels in eval_loader:
    logits = model(input_ids, masked_pos)  # [batch_size, num_masked, vocab_size]
    
    # top 10
    top_k_preds = torch.topk(logits, k=10, dim=-1).indices
    
    for batch_idx in range(len(input_ids)):
        recommended_items = top_k_preds[batch_idx].cpu().numpy()  # top-10
        relevant_items = masked_labels[batch_idx].cpu().numpy()  
        
        # Recall@10
        recall_sum += recall_at_k(recommended_items, relevant_items, k=10)
        
        # NDCG@10
        ndcg_sum += ndcg_at_k(recommended_items, relevant_items, k=10)
        
        num_samples += 1

# average Recall@10 and NDCG@10
average_recall = recall_sum / num_samples
average_ndcg = ndcg_sum / num_samples

print(f"Average Recall@10: {average_recall:.4f}")
print(f"Average NDCG@10: {average_ndcg:.4f}")