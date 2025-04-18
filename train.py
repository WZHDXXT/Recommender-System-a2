from bert4rec import BERT4Rec  
from maskdataset import BERT4RecDataset, collate_fn, EvalDataset
from torch.utils.data import DataLoader
from data_load import data_load
import torch.nn as nn
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from evaluation import recall_at_k, ndcg_at_k

user_train, user_valid, user_test, usernum, itemnum = data_load('ratings.dat')
epochs = 50
max_len = 20
vocab_size = itemnum + 2  # +1 for [MASK]
mask_token = vocab_size - 1  # index of [MASK], no conflicting with existing item ids
model = BERT4Rec(vocab_size=vocab_size, max_seq_length=max_len)
model.train()

# training set
train_dataset = BERT4RecDataset(user_train, max_len=max_len, mask_token=mask_token)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

# validation set
valid_dataset = EvalDataset(user_valid, max_len=max_len, mask_token=mask_token)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

from collections import Counter

all_items = [movie for user_seq in user_train.values() for movie in user_seq if movie != 0]
freq = Counter(all_items)
top_items = freq.most_common(20)
print("Top frequent items:", top_items)


for epoch in range(epochs):
    model.train()
    for batch_idx, (input_ids, masked_pos, masked_labels) in enumerate(train_loader):
        logits = model(input_ids, masked_pos)

        # Print the shape of logits and the values from the first batch
        # print(f"Epoch {epoch+1}, Batch {batch_idx+1}: logits shape: {logits.shape}")
        # print(f"Logits (first batch): {logits[0]}")

        loss = criterion(logits.view(-1, vocab_size), masked_labels.view(-1).long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    # validation monitoring on NDCG@10
    model.eval()
    val_recall_sum = 0
    val_ndcg_sum = 0
    val_num_samples = 0

    with torch.no_grad():
        for batch_idx, (input_ids, masked_pos, masked_labels) in enumerate(valid_loader):
            logits = model(input_ids, masked_pos)

            # print(f"Epoch {epoch+1}, Batch {batch_idx+1}: logits shape: {logits.shape}")
            # print(f"Logits (first batch): {logits[0]}")

            # top 10
            logits[:, :, 0] = -float('inf')  # shield padding token
            logits[:, :, mask_token] = -float('inf')  # shield mask token
            top_k_preds = torch.topk(logits, k=10, dim=-1).indices
            # print(f"Top-10 predictions (first batch): {top_k_preds[0]}")  # Print top-10 predictions for the first sample
            
            for batch_idx in range(len(input_ids)):
                recommended_items = top_k_preds[batch_idx].view(-1).cpu().numpy()  # top-10
                # relevant_items = input_ids[batch_idx].clone().cpu().numpy()
                # relevant_items[masked_pos[batch_idx].cpu().numpy()] = masked_labels[batch_idx].cpu().numpy()
                # relevant_items = [item for item in relevant_items if item != 0 and item != mask_token]
                # relevant_items = [item for item in masked_labels[batch_idx].tolist() if item != 0 and item != mask_token]                
                relevant_items = [masked_labels[batch_idx].item()]
                print(f"Recommended items (top-10): {recommended_items}")
                print(f"Relevant items (before filtering): {relevant_items}")
                
                # Recall@10
                recall = recall_at_k(recommended_items, relevant_items, k=10)
                print(f"Recall@10 (batch {batch_idx + 1}): {recall}")
                val_recall_sum += recall
                
                # NDCG@10
                ndcg = ndcg_at_k(recommended_items, relevant_items, k=10)
                print(f"NDCG@10 (batch {batch_idx + 1}): {ndcg}")
                val_ndcg_sum += ndcg
                
                val_num_samples += 1
        
    average_recall = val_recall_sum / val_num_samples
    average_ndcg = val_ndcg_sum / val_num_samples
    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Valid NDCG: {average_ndcg:.4f}")
    print(f"LR: {scheduler.get_last_lr()[0]}")
    scheduler.step()

# only mask's last position
eval_dataset = EvalDataset(user_test, max_len=max_len, mask_token=mask_token)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn)

# evaluation on test set
model.eval() 
recall_sum = 0
ndcg_sum = 0
num_samples = 0

for input_ids, masked_pos, masked_labels in eval_loader:
    logits = model(input_ids, masked_pos)  # [batch_size, num_masked, vocab_size]
    
    # top 10
    logits[:, :, 0] = -float('inf')  # shield padding token
    logits[:, :, mask_token] = -float('inf')  # shield mask token
    top_k_preds = torch.topk(logits, k=10, dim=-1).indices
    
    # print the first sample's top-10 predictions
    # print(f"Top-10 predictions (first sample): {top_k_preds[0]}")
    
    for batch_idx in range(len(input_ids)):
        recommended_items = top_k_preds[batch_idx].view(-1).cpu().numpy()  # top-10
        
        # relevant_items = input_ids[batch_idx].clone().cpu().numpy()
        # relevant_items[masked_pos[batch_idx].cpu().numpy()] = masked_labels[batch_idx].cpu().numpy()
        # relevant_items = [item for item in relevant_items if item != 0 and item != mask_token]
        relevant_items = [masked_labels[batch_idx].item()]
        
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
