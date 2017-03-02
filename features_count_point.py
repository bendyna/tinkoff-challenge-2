#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm

train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

columns = [
    'count_point_all',                  # количество транзакций с такой точкой среди всех мерчантов
                                        # (надо пересчитывать при кросс-валидации)
    'count_point_mid',                  # количество транзакций с такой точкой у этого мерчанта
    'count_point_mid_unique_time',      # количество транзакций с такой точкой у этого мерчанта и с
                                        # уникальным временем транзакции
]

train_ids = set(train_data.mid)

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

    vc_local = mid_trans.latlon.value_counts()
    array[index:index + size, 0] = mid_trans.latlon.apply(lambda ll: vc_all[ll])
    array[index:index + size, 1] = mid_trans.latlon.apply(lambda ll: vc_local[ll])
    array[index:index + size, 2] = mid_trans.latlon.apply(
        lambda ll: mid_trans[mid_trans.latlon == ll].ttime.value_counts().shape[0])


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_count_point_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
