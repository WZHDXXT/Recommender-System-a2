import pandas as pd
import os
from collections import defaultdict
import numpy as np

class MakeSequenceDataSet():
    # SequenceDataSet for MovieLens 1M
    def __init__(self, data_path):
        print('Loading data...')
        self.df = pd.read_csv(
            os.path.join(data_path, 'ratings.dat'),
            sep='::',
            engine='python',
            names=['userId', 'movieId', 'rating', 'timestamp']
        )

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder(self.df['movieId'])
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder(self.df['userId'])
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['movieId'].apply(lambda x: self.item_encoder[x] + 1)  # +1 for padding=0
        self.df['user_idx'] = self.df['userId'].apply(lambda x: self.user_encoder[x])

        # Sort interactions by user and timestamp
        self.df = self.df.sort_values(['user_idx', 'timestamp'])
        self.user_train, self.user_valid, self.user_test = self.generate_sequence_data()

        print('Finish')

    def generate_encoder_decoder(self, col):
        encoder = {}
        decoder = {}
        ids = col.unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def generate_sequence_data(self):
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}

        # Group by user and build sequences
        group_df = self.df.groupby('user_idx')
        for user, group in group_df:
            seq = group['item_idx'].tolist()
            if len(seq) < 5:  # Ensure there's enough history
                continue
            # seq = seq[-20:]  # Limit sequence to the last 20 items
            np.random.shuffle(seq)

            # Split data into 70% for training, 15% for validation, 15% for testing
            train_size = int(len(seq) * 0.7)
            valid_size = int(len(seq) * 0.15)

            user_train[user] = seq[:train_size]
            user_valid[user] = seq[train_size:train_size + valid_size]
            user_test[user] = seq[train_size + valid_size:]

        return user_train, user_valid, user_test

    def get_train_valid_data(self):
        return self.user_train, self.user_valid, self.user_test
    

# def main():
#     dataset = MakeSequenceDataSet(data_path='./')
#     user_train, user_valid, user_test = dataset.get_train_valid_data()
#     num_user = dataset.num_user
#     num_item = dataset.num_item
#     print("User 0 train sequence:", user_train[0])
#     print("User 0 validation item:", user_valid[0])
#     print("User 0 test item:", user_test[0])
# main()