#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm


def is_last_record(trans):
    res = []
    for i, row in trans.iterrows():
        res.append(row.rectime == trans[trans.ttime == row.ttime].rectime.max())
    return res

train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

# исходим из предположения, что если у мерчанта есть несколько записей с одним временем транзакции,
# то это одна транзакция с несколькими записями
columns = [
    'record_count',         # количество транзакций разных по времени у мерчанта
    'latlon_count',         # количество разных координат среди записей этой же транзакции
    'is_last_record',       # является ли запись транзакции последней по времени среди всех записей этой транзакции
]

train_ids = list(train_data.mid)

count_all_ids = transactions.mid.unique().shape[0]

train_array = np.zeros((200000, len(columns)))
train_index = 0
test_array = np.zeros((200000, len(columns)))
test_index = 0

transactions['latlon'] = transactions.lat.apply(str) + '_' + transactions.lon.apply(str)
vc_all = transactions['latlon'].value_counts()

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

    vc_local = mid_trans.ttime.value_counts()
    array[index:index + size, 0] = mid_trans.ttime.apply(lambda t: vc_local[t])
    array[index:index + size, 1] = mid_trans.ttime.apply(
        lambda t: mid_trans[mid_trans.ttime == t].latlon.value_counts().shape[0])
    array[index:index + size, 2] = is_last_record(mid_trans)


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_transaction_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
