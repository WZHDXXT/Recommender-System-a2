from collections import defaultdict

def data_load(filename):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    # assume user/item index starting from 1
    with open(filename, 'r', encoding="utf-8") as file:
        for line in file:
            user_id, movie_id, rating, timestamp = map(int, line.strip().split("::"))
            usernum = max(user_id, usernum)
            itemnum = max(movie_id, itemnum)
            if rating > 3:
                User[user_id].append((timestamp, movie_id))

    User = {user: items for user, items in User.items() if len(items) >= 5}

    for user in User:
        User[user].sort()
        User[user] = [movie_id for _, movie_id in User[user]]

    max_len = 20
    for user in User:
        seq = User[user][-max_len:] 
        if len(seq) < max_len:
            seq = [0] * (max_len - len(seq)) + seq  
        user_train[user] = seq[:-2]
        user_valid[user] = [seq[-2]]
        user_test[user] = [seq[-1]]
    return user_train, user_valid, user_test, usernum, itemnum


def main():
    user_train, user_valid, user_test, usernum, itemnum = data_load('ratings.dat')
    print(user_train)

main()
