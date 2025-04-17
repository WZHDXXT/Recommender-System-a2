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
vocab_size = itemnum + 2  # +1 for padding, +1 for [MASK]
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
optimizer = optim.Adam(model.parameters(), lr=5e-5)
# scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    model.train()
    for input_ids, masked_pos, masked_labels in train_loader:
        logits = model(input_ids, masked_pos)
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
        for input_ids, masked_pos, masked_labels in valid_loader:
            logits = model(input_ids, masked_pos)  # [batch_size, num_masked, vocab_size]
    
            # top 10
            logits[:, :, 0] = -float('inf')  # shield padding token
            logits[:, :, mask_token] = -float('inf')  # shield mask token
            top_k_preds = torch.topk(logits, k=10, dim=-1).indices
            
            for batch_idx in range(len(input_ids)):
                recommended_items = top_k_preds[batch_idx].view(-1).cpu().numpy()  # top-10
                relevant_items = masked_labels[batch_idx].cpu().numpy()          
                relevant_items = relevant_items[relevant_items != 0]  # shield padding 
                
                # Recall@10
                val_recall_sum += recall_at_k(recommended_items, relevant_items, k=10)        
                
                # NDCG@10
                val_ndcg_sum += ndcg_at_k(recommended_items, relevant_items, k=10)        
                val_num_samples += 1
                # if epoch % 5 == 0 and batch_idx == 0:
                #     print("Input IDs:", input_ids[:4])
                # if epoch % 5 == 0 and batch_idx == 0:
                #     print("==== DEBUG (validation sample) ====")
                #     print("Predicted top-10:", recommended_items.tolist())
                #     print("Ground truth:", relevant_items.tolist())
                # if epoch % 5 == 0 and batch_idx == 0:
                #     logit_values = logits[batch_idx, 0].detach().cpu().numpy()
                #     top10_scores = logit_values[top_k_preds[batch_idx].cpu().numpy()]
                #     print("Top-10 logits scores:", top10_scores.tolist())
    
    average_recall = val_recall_sum / val_num_samples
    average_ndcg = val_ndcg_sum / val_num_samples
    print(f"Epoch {epoch+1}, Train Loss: {loss.item():.4f}, Valid NDCG: {average_ndcg:.4f}")
    # print(f"LR: {scheduler.get_last_lr()[0]}")
    # scheduler.step()

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
    
    for batch_idx in range(len(input_ids)):
        recommended_items = top_k_preds[batch_idx].view(-1).cpu().numpy()  # top-10
        relevant_items = masked_labels[batch_idx].cpu().numpy()          
        relevant_items = relevant_items[relevant_items != 0]  # shield padding 
        
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
