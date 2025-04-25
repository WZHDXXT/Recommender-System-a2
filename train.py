from model1 import Bert4Rec
from torch.utils.data import DataLoader
from maskdataset import BERTRecDataSet
from data_load import MakeSequenceDataSet
import torch
import torch.nn as nn
from evaluation import evaluate, evaluate_test, full_ranking_evaluate_with_validation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyperparameter
max_len = 20
mask_prob = 0.15

dataset = MakeSequenceDataSet(data_path='./')
num_user = dataset.num_user
num_item = dataset.num_item
user_train, user_valid, user_test = dataset.get_train_valid_data()
bert4rec_dataset = BERTRecDataSet(user_train, max_len, num_user, num_item, mask_prob)

data_loader = DataLoader(
    bert4rec_dataset, 
    batch_size = 32, 
    shuffle = True, 
    pin_memory = True,
    num_workers = 0
)



# model = BERT(
#     max_seq_length=20,
#     vocab_size=3706,      
#     bert_num_blocks=2,
#     bert_num_heads=2,
#     hidden_size=128,
#     bert_dropout=0.1
# )

vocab_size = num_item  
max_seq_length = 20

# hyperparameter
bert_num_blocks = 2
bert_num_heads = 2
hidden_size = 512

bert_dropout = 0.1

model = Bert4Rec(
    max_seq_length=max_seq_length,
    vocab_size=vocab_size,
    bert_num_blocks=bert_num_blocks,
    bert_num_heads=bert_num_heads,
    hidden_size=hidden_size,
    bert_dropout=bert_dropout
)
model.to(device)


def train(model, criterion, optimizer, data_loader):
    metrics = {
    'train_loss': [],
}
    model.train()
    loss_val = 0
    i = len(metrics['train_loss'])

    for seq, labels in data_loader:
        seq, labels = seq.to(device), labels.to(device)
        logits = model(seq)  # [batch, seq_len, vocab_size]
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss_val += loss.item()
        metrics['train_loss'].append((i, loss.item()))

        loss.backward()
        optimizer.step()
        i += 1

    loss_val /= len(data_loader)
    return loss_val

# ignore padding
criterion = nn.CrossEntropyLoss(ignore_index=0) 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_list = []
ndcg_list = []
best_ndcg = 0
counter = 0
epoch_num = 20
patience = 10
for epoch in range(1, epoch_num+1):
    train_loss = train(
        model = model, 
        criterion = criterion, 
        optimizer = optimizer, 
        data_loader = data_loader)
    loss_list.append(train_loss)

    print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}')

    ndcg, recall = evaluate(
        model = model, 
        user_train = user_train,
        user_valid = user_valid, 
        max_len = max_len,
        make_sequence_dataset = dataset,
        bert4rec_dataset = bert4rec_dataset,
        K = 10
    )
    print(ndcg, recall)
    ndcg_list.append(ndcg)

    # early stopping with ndcg
    if ndcg > best_ndcg:
        counter = 0
        best_ndcg = ndcg
    else:
        counter += 1
        print(f"Early Stopping Counter: {counter}/{patience}")
        if counter >= patience:
            print("Early stopping!")
            break


# ndcg_test, recall_test = full_ranking_evaluate_with_validation(
#     model=model,
#     user_train=user_train,
#     user_valid=user_valid,
#     user_test=user_test,
#     max_len=max_len,
#     vocab_size=vocab_size,
#     device=device,
#     K=10
# )

ndcg_test, recall_test = evaluate_test(
    model=model,
    user_train=user_train,
    user_valid=user_valid,
    user_test=user_test,
    max_len = max_len,
    make_sequence_dataset = dataset,
    bert4rec_dataset = bert4rec_dataset,
    K = 10
)
print(f"Final Test Recall@10: {recall_test:.4f}")
print(f"Final Test NDCG@10: {ndcg_test:.4f}")
