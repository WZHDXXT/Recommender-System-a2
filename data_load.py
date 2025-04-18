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
        self.df['user_idx'] = pd.factorize(self.df['userId'])[0]
        self.df['item_idx'] = pd.factorize(self.df['movieId'])[0]
        self.num_item = self.df['movieId'].nunique()
        self.num_user = self.df['userId'].nunique()

        # Sort interactions by user and timestamp
        self.df = self.df.sort_values(['user_idx', 'timestamp'])

        # print('Generating sequence data...')
        self.user_train, self.user_valid, self.user_test = self.generate_sequence_data()

        print('Finish')

    

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
            # keep users with at least five feedbacks
            if len(seq) < 5:
                continue  
            # setting train, valid and test
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user] = [seq[-1]]
        return user_train, user_valid, user_test

    def get_train_valid_data(self):
        """Return the processed train and valid dictionaries."""
        return self.user_train, self.user_valid, self.user_test
    

# def main():
#     dataset = MakeSequenceDataSet(data_path='./')
#     user_train, user_valid, user_test = dataset.get_train_valid_data()
#     num_user = dataset.num_user
#     num_item = dataset.num_item
#     print(num_user, num_item)
#     print("User 0 train sequence:", user_train[0])
#     print("User 0 validation item:", user_valid[0])
#     print("User 0 test item:", user_test[0])
# main()