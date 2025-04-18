import pandas as pd
import os
from collections import defaultdict

class MakeSequenceDataSet():
    """
    SequenceDataSet for MovieLens 1M

    Converts the original ratings.dat (user::movie::rating::timestamp)
    into a format suitable for sequence-based recommendation (e.g., BERT4Rec),
    generating per-user sequences split into train and validation sets.
    """

    def __init__(self, data_path):
        print('Reading ratings.dat...')
        self.df = pd.read_csv(
            os.path.join(data_path, 'ratings.dat'),
            sep='::',
            engine='python',
            names=['userId', 'movieId', 'rating', 'timestamp']
        )

        print('Generating ID encoders...')
        self.item_encoder, self.item_decoder = self.generate_encoder_decoder(self.df['movieId'])
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder(self.df['userId'])
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        # Encode movieId and userId to indices
        self.df['item_idx'] = self.df['movieId'].apply(lambda x: self.item_encoder[x] + 1)  # +1 for padding=0
        self.df['user_idx'] = self.df['userId'].apply(lambda x: self.user_encoder[x])

        # Sort interactions by user and timestamp
        self.df = self.df.sort_values(['user_idx', 'timestamp'])

        print('Generating sequence data...')
        self.user_train, self.user_valid, self.user_test = self.generate_sequence_data()

        print('Finish')

    def generate_encoder_decoder(self, col):
        """Generate mapping from original IDs to indices and vice versa."""
        encoder = {}
        decoder = {}
        ids = col.unique()

        for idx, _id in enumerate(ids):
            encoder[_id] = idx
            decoder[idx] = _id

        return encoder, decoder

    def generate_sequence_data(self):
        """
        Convert raw interactions into user-wise sequences.
        Each user: use all but last item for training, last for validation.
        """
        users = defaultdict(list)
        user_train = {}
        user_valid = {}
        user_test = {}

        # Group by user and build sequences
        group_df = self.df.groupby('user_idx')
        for user, group in group_df:
            seq = group['item_idx'].tolist()
            if len(seq) < 5:
                continue  # skip users with too short history
            user_train[user] = seq[:-2]
            user_valid[user] = seq[:-1]
            user_test[user] = seq[1:]
        return user_train, user_valid, user_test

    def get_train_valid_data(self):
        """Return the processed train and valid dictionaries."""
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