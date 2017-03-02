#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm


def perc_rect(trans, dist, is_unique):
    count_trans = trans.shape[0]
    latlon = zip(trans.lat.values, trans.lon.values)
    count_div = count_trans
    if is_unique:
        count_div = len(set(latlon))
    if count_div <= 1:
        return 0, 0
    res = [0] * count_trans
    for i in xrange(count_trans):
        st = set()
        for j in xrange(count_trans):
            if j == i:
                continue
            if is_unique and latlon[i][0] == latlon[j][0] and latlon[i][1] == latlon[j][1]:
                continue
            if abs(latlon[i][0] - latlon[j][0]) < dist and abs(latlon[i][1] - latlon[j][1]) < dist:
                key = (latlon[i], latlon[j])
                if is_unique and key in st:
                    continue
                st.add(key)
                res[i] += 1
    return res, [t * 1.0 / (count_div - 1) for t in res]

train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

columns = [
    'count_rect_002',           # количество транзакций этого же мерчанта,
                                # которые в окрестности 0.002 от текущей транзакции
    'perc_rect_002',            # процент транзакций этого же мерчанта,
                                # которые в окрестности 0.002 от текущей транзакции
    'count_rect_002_unique',    # количество уникальных транзакций этого же мерчанта,
                                # которые в окрестности 0.002 от текущей транзакции
    'perc_rect_002_unique',     # процент уникальных транзакций этого же мерчанта,
                                # которые в окрестности 0.002 от текущей транзакции
    'count_rect_004',
    'perc_rect_004',
    'count_rect_004_unique',
    'perc_rect_004_unique',
    'count_rect_01',
    'perc_rect_01',
    'count_rect_01_unique',
    'perc_rect_01_unique',
    'count_rect_02',
    'perc_rect_02',
    'count_rect_02_unique',
    'perc_rect_02_unique',
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

    for i, d in enumerate([0.002, 0.004, 0.01, 0.02]):
        count, perc = perc_rect(mid_trans, d, False)
        array[index:index + size, i * 4 + 0] = count
        array[index:index + size, i * 4 + 1] = perc
        count, perc = perc_rect(mid_trans, d, True)
        array[index:index + size, i * 4 + 2] = count
        array[index:index + size, i * 4 + 3] = perc


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_perc_rect_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
