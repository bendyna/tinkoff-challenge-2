#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import time


def time_diff(t1, t2):
    t = time.mktime(time.strptime(t2, "%H:%M:%S")) - time.mktime(time.strptime(t1, "%H:%M:%S"))
    if t < -80000:
        t += 86400
    if t >= 80000:
        t = 86400 - t
    return t


def get_seconds(t):
    h, m, s = t.split(':')
    return 3600 * int(h) + 60 * int(m) + int(s)


train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

columns = [
    'time_diff',    # разница в секундах между временем транзакции и временем записи
    'time_sec',     # секунда в сутках времени транзакции (0-86400)
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

    array[index:index + size, 0] = mid_trans.apply(lambda x: time_diff(x.ttime, x.rectime), axis=1)
    array[index:index + size, 1] = mid_trans.apply(lambda x: get_seconds(x.ttime), axis=1)


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_time_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
