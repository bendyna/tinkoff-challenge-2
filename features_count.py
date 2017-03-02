#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

columns = [
    'count_good',           # количество транзакций у мерчанта
    'count_good_unique',    # количество уникальных транзакций у мерчанта
]

train_ids = set(train_data.mid)

count_all_ids = transactions.mid.unique().shape[0]

train_array = np.zeros((200000, len(columns)))
train_index = 0
test_array = np.zeros((200000, len(columns)))
test_index = 0

for merchant_id in tqdm(transactions.mid.unique(), total=count_all_ids):
    mid_transactions = transactions[transactions.mid == merchant_id]

    count_good = mid_transactions.shape[0]

    if merchant_id in train_ids:
        array = train_array
        index = train_index
        train_index += count_good
    else:
        array = test_array
        index = test_index
        test_index += count_good

    array[index:index + count_good, 0] = count_good
    unique = (mid_transactions.lat.apply(str) + mid_transactions.lon.apply(str)).value_counts().shape[0]
    array[index:index + count_good, 1] = unique


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_count_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
