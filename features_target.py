#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

HSIDE = 0.002


def is_close(trans):
    mid = trans.mid.iloc[0]
    if mid not in train_ids:
        return 0
    row = train_data[train_data.mid == mid]
    lat = row.lat.iloc[0]
    lon = row.lon.iloc[0]
    return (trans.lat > lat - HSIDE) & (trans.lat < lat + HSIDE) & \
           (trans.lon > lon - HSIDE) & (trans.lon < lon + HSIDE)


train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

columns = [
    'target',
]

train_ids = list(train_data.mid)

count_all_ids = transactions.mid.unique().shape[0]

train_array = np.zeros((200000, len(columns)))
train_index = 0
test_array = np.zeros((200000, len(columns)))
test_index = 0

for merchant_id in tqdm(transactions.mid.unique(), total=count_all_ids):
    mid_trans = transactions[transactions.mid == merchant_id]

    size = mid_trans.shape[0]

    if size == 0:
        continue

    if merchant_id in train_ids:
        array = train_array
        index = train_index
        train_index += size
    else:
        array = test_array
        index = test_index
        test_index += size

    array[index:index + size, 0] = is_close(mid_trans)


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_target_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
