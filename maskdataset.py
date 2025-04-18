from torch.utils.data import Dataset
import torch
import numpy as np
import random
from data_load import MakeSequenceDataSet

class BERTRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, num_item, mask_prob):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self.mask_prob = mask_prob

    def __len__(self):
        return self.num_user

    def __getitem__(self, user): 
        user_seq = self.user_train[user]
        tokens = []
        labels = []

        rated_items = user_seq[-self.max_len:]  

        for s in rated_items:
            prob = np.random.random()
            if prob < self.mask_prob:
                prob /= self.mask_prob
                if prob < 0.8:
                    tokens.append(self.num_item + 1)  # mask token
                elif prob < 0.9:
                    rnd_token = self.random_neg_sampling(rated_items, 1)[0]  # negative
                    tokens.append(rnd_token)
                else:
                    tokens.append(s) 
            else:
                tokens.append(s) 
            labels.append(s)

        # padding
        pad_len = self.max_len - len(tokens)
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

    def random_neg_sampling(self, rated_items, num_samples):
        available_items = set(range(1, self.num_item + 1)) - set(rated_items)
        return random.sample(available_items, num_samples)
    


# def main():
#     max_len = 20
#     mask_prob = 0.15
#     dataset = MakeSequenceDataSet(data_path='./')
#     num_user = dataset.num_user
#     num_item = dataset.num_item
#     user_train, user_valid, user_test = dataset.get_train_valid_data()
#     bert4rec_dataset = BERTRecDataSet(user_train, max_len, num_user, num_item, mask_prob)
    
#     tokens, labels = bert4rec_dataset[0]
#     print("Tokens for user 0:", tokens)
#     print("Labels for user 0:", labels)

# main()