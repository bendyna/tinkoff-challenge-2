#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
from math import sqrt


def get_percentile_uniq(arr):
    arr_uniq = list(set(arr))
    if len(arr_uniq) > 1:
        arr_uniq = sorted(enumerate(arr_uniq), key=lambda x: x[1])
        dict_uniq = {x[1]: k * 1.0 / (len(arr_uniq) - 1) for k, x in enumerate(arr_uniq)}
        res = [dict_uniq[x] for x in arr]
    else:
        res = [0]
    return res


train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

# Все расстояния тут считаются не в метрах, а в единицах геокоординат
# несмотря на то что градусы по долготе и широте отличаются в метрах
# Для каждой транзакции признаки считаются только по тем транзакциям, которые есть у этого мерчанта
columns = [
    'mean_dist',                    # среднее расстояние до остальных точек мерчанта
    'mean_dist_unique',             # среднее расстояние до уникальных остальных точек мерчанта
    'mean_dist_top_50',             # среднее расстояние до ближайших 50% остальных точек мерчанта
    'mean_dist_top_50_unique',      # среднее расстояние до ближайших 50% уникальных остальных точек мерчанта
    'mean_dist_percentile',         # mean_dist_top_50 только не абсолютное значение, а перцентиль среди
                                    # всех транзакций мерчанта
    'mean_dist_percentile_unique',  # mean_dist_top_50_unique только не абсолютное значение, а перцентиль среди
                                    # всех транзакций мерчанта
    'dist_nearest',                 # расстояние до ближайшей транзакции (может быть 0)
    'dist_nearest_uniq',            # расстояние до ближайшей транзакции (ненулевое)
    'dist_10_perc',                 # 10ый перцентиль в массиве расстояния до остальных транзакций этого мерчанта
    'dist_10_perc_uniq',            # 10ый перцентиль в массиве расстояния до остальных уникальных транзакций
                                    # этого мерчанта
    'dist_20_perc',
    'dist_20_perc_uniq',
    'dist_40_perc',
    'dist_40_perc_uniq',
    'perc_lat_uniq',                # перцентиль по широте среди всех уникальных транзакций этого мерчанта
    'perc_lon_uniq',                # перцентиль по долготе среди всех уникальных транзакций этого мерчанта
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

    latlon = [(mid_trans.iloc[j, :].lat, mid_trans.iloc[j, :].lon) for j in xrange(size)]
    mean_dist = [0] * size
    mean_dist_top_50 = [0] * size
    dist_nearest = [0] * size
    dist_perc10 = [0] * size
    dist_perc20 = [0] * size
    dist_perc40 = [0] * size
    for i in xrange(size):
        distances = []
        for j in xrange(size):
            if j == i:
                continue
            distances.append(sqrt((latlon[i][0] - latlon[j][0]) ** 2 + (latlon[i][1] - latlon[j][1]) ** 2))
        mean_dist[i] = sum(distances)
        sorted_distances = sorted(distances)
        mean_dist_top_50[i] = sum(sorted_distances[:(size - 1) / 2]) if size >= 3 else 0
        dist_nearest[i] = sorted_distances[0] if size > 1 else 0
        dist_perc10[i] = sorted_distances[int((size - 2) * 0.1)] if size > 1 else 0
        dist_perc20[i] = sorted_distances[int((size - 2) * 0.2)] if size > 1 else 0
        dist_perc40[i] = sorted_distances[int((size - 2) * 0.4)] if size > 1 else 0
    if size > 1:
        mean_dist = [t * 1.0 / (size - 1) for t in mean_dist]

    array[index:index + size, 0] = mean_dist
    array[index:index + size, 2] = mean_dist_top_50
    array[index:index + size, 4] = get_percentile_uniq(mean_dist_top_50)
    array[index:index + size, 6] = dist_nearest
    array[index:index + size, 8] = dist_perc10
    array[index:index + size, 10] = dist_perc20
    array[index:index + size, 12] = dist_perc40

    count_uniq = len(set(latlon))
    mean_dist_uniq = [0] * size
    mean_dist_top_50_uniq = [0] * size
    dist_nearest_uniq = [0] * size
    dist_perc10_uniq = [0] * size
    dist_perc20_uniq = [0] * size
    dist_perc40_uniq = [0] * size
    for i in xrange(size):
        distances = []
        st = set()
        for j in xrange(size):
            if j == i:
                continue
            if latlon[i][0] == latlon[j][0] and latlon[i][1] == latlon[j][1]:
                continue
            key = (latlon[i], latlon[j])
            if key in st:
                continue
            st.add(key)
            distances.append(sqrt((latlon[i][0] - latlon[j][0]) ** 2 + (latlon[i][1] - latlon[j][1]) ** 2))
        mean_dist_uniq[i] = sum(distances)
        sorted_distances = sorted(distances)
        mean_dist_top_50_uniq[i] = sum(sorted_distances[:(count_uniq - 1) / 2]) if count_uniq >= 3 else 0
        dist_nearest_uniq[i] = sorted_distances[0] if count_uniq > 1 else 0
        dist_perc10[i] = sorted_distances[int((count_uniq - 2) * 0.1)] if count_uniq > 1 else 0
        dist_perc20[i] = sorted_distances[int((count_uniq - 2) * 0.2)] if count_uniq > 1 else 0
        dist_perc40[i] = sorted_distances[int((count_uniq - 2) * 0.4)] if count_uniq > 1 else 0
    if count_uniq > 1:
        mean_dist_uniq = [t * 1.0 / (count_uniq - 1) for t in mean_dist_uniq]

    array[index:index + size, 1] = mean_dist_uniq
    array[index:index + size, 3] = mean_dist_top_50_uniq
    array[index:index + size, 5] = get_percentile_uniq(mean_dist_top_50_uniq)
    array[index:index + size, 7] = dist_nearest_uniq
    array[index:index + size, 9] = dist_perc10
    array[index:index + size, 11] = dist_perc20
    array[index:index + size, 13] = dist_perc40

    array[index:index + size, 14] = get_percentile_uniq([t[0] for t in latlon])
    array[index:index + size, 15] = get_percentile_uniq([t[1] for t in latlon])


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_percentile_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
