import random
from torch.utils.data import Dataset
import torch

class BERT4RecDataset(Dataset):
    def __init__(self, user_train, max_len=20, mask_token=9999):
        self.users = list(user_train.keys())
        self.user_train = user_train
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.user_train[user]
        
        # Debug print: Check the original sequence
        # print(f"User {user} original sequence:", seq)
        
        masked_seq, masked_pos, masked_labels = mask_sequence(seq, self.mask_token, max_len=self.max_len)

        # Debug print: Check the masked sequence, positions, and labels
        # print(f"User {user} masked sequence:", masked_seq)
        # print(f"User {user} masked positions:", masked_pos)
        # print(f"User {user} masked labels:", masked_labels)

        return torch.tensor(masked_seq), torch.tensor(masked_pos), torch.tensor(masked_labels)
    
def collate_fn(batch):
    input_ids, masked_pos, masked_labels = zip(*batch)
    max_len = max(pos.size(0) for pos in masked_pos)
    padded_pos = [torch.cat([pos, torch.zeros(max_len - pos.size(0), dtype=torch.long)]) for pos in masked_pos]
    padded_labels = [torch.cat([label, torch.zeros(max_len - label.size(0), dtype=torch.long)]) for label in masked_labels]
    return (
        torch.stack(input_ids),            # [batch_size, seq_len]
        torch.stack(padded_pos),           # [batch_size, max_num_masked]
        torch.stack(padded_labels)         # [batch_size, max_num_masked]
    )

class EvalDataset(Dataset):
    def __init__(self, user_sequences, max_len=20, mask_token=9999):
        self.users = list(user_sequences.keys())
        self.sequences = user_sequences
        self.max_len = max_len
        self.mask_token = mask_token

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        user = self.users[idx]
        seq = self.sequences[user][-self.max_len:]  

        pad_len = self.max_len - len(seq)
        input_seq = [0] * pad_len + seq

        # Mask the last position
        masked_seq = input_seq[:-1] + [self.mask_token]
        masked_pos = [self.max_len - 1]
        masked_label = [input_seq[-1]]   

        return torch.tensor(masked_seq), torch.tensor(masked_pos), torch.tensor(masked_label)
    

def mask_sequence(seq, mask_token=9999, mask_ratio=0.5, max_len=20):
    seq = seq[-max_len:]
    num_to_mask = max(1, int(len(seq) * mask_ratio))
    mask_indices = random.sample(range(len(seq)), num_to_mask)

    masked_seq = []
    masked_pos = []
    masked_labels = []

    for idx, item in enumerate(seq):
        if item == 0:
            masked_seq.append(0)
            continue
        if idx in mask_indices:
            masked_seq.append(mask_token)
            masked_pos.append(idx)
            masked_labels.append(item)
        else:
            masked_seq.append(item)

    pad_len = max_len - len(masked_seq)
    masked_seq = [0] * pad_len + masked_seq
    masked_pos = [pos + pad_len for pos in masked_pos]  # shift position
    return masked_seq, masked_pos, masked_labels