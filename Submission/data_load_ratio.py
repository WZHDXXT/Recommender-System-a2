import pandas as pd
import os
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
        # Treat ratings ≥ 4 as positive interactions.
        self.df = self.df[self.df['rating'] >= 4] 

        self.item_encoder, self.item_decoder = self.generate_encoder_decoder(self.df['movieId'])
        self.user_encoder, self.user_decoder = self.generate_encoder_decoder(self.df['userId'])
        self.num_item, self.num_user = len(self.item_encoder), len(self.user_encoder)

        self.df['item_idx'] = self.df['movieId'].apply(lambda x: self.item_encoder[x] + 1)  # +1 for padding=0
        self.df['user_idx'] = self.df['userId'].apply(lambda x: self.user_encoder[x])
        
        # chronologically ordered sequence
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
            
            # Filter out users with fewer than 5 interactions.
            if len(seq) < 5:  
                continue
            # seq = seq[-20:]
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
    

def main():
    dataset = MakeSequenceDataSet(data_path='./')
    user_train, user_valid, user_test = dataset.get_train_valid_data()
    total_lengths = [len(user_train[u]) + len(user_valid[u]) + len(user_test[u]) for u in user_train]

    # Print some stats
    print(f"Average total sequence length: {sum(total_lengths) / len(total_lengths):.2f}")
    print(f"Min total sequence length: {min(total_lengths)}")
    print(f"Max total sequence length: {max(total_lengths)}")

    # Plot the distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(total_lengths, bins=30, kde=True, color='skyblue', edgecolor='black')
    plt.xlabel('Total sequence length per user', fontsize=12, fontweight='bold')
    plt.ylabel('Number of users', fontsize=12, fontweight='bold')
    plt.title('Distribution of Total Sequence Lengths', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Save the figure
    plt.savefig('train_interaction_distribution.png', bbox_inches='tight', dpi=300)

    plt.show()
    total_lengths = [len(user_train[u]) + len(user_valid[u]) + len(user_test[u]) for u in user_train]

    # Print some stats
    print(f"Average total sequence length: {sum(total_lengths) / len(total_lengths):.2f}")
    print(f"Min total sequence length: {min(total_lengths)}")
    print(f"Max total sequence length: {max(total_lengths)}")

    # Filter only users with <= 50 interactions
    filtered_total_lengths = [l for l in total_lengths if l <= 40]

    print(f"Number of users with <=50 total sequence length: {len(filtered_total_lengths)}")

    # Plot the distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(filtered_total_lengths, bins=50, kde=True, color='skyblue', edgecolor='black')
    plt.xlabel('Total sequence length per user', fontsize=12, fontweight='bold')
    plt.ylabel('Number of users', fontsize=12, fontweight='bold')
    plt.title('Distribution of Total Sequence Lengths (≤50)', fontsize=14, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Save the figure
    plt.savefig('train_interaction_distribution_below_50.png', bbox_inches='tight', dpi=300)

    plt.show()
# main()