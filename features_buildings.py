#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tqdm import tqdm
import buildings_lib

train_data = pd.read_csv('data/mtrain.csv')
transactions = pd.read_csv('data/transactions_good.csv')

columns = [
    'is_in_building',       # находится ли точка транзакции внутри здания
    'is_shop_mall',         # находится ли точка в торговом центре
    'area',                 # площадь здания, в котором находится точка (или 0, если не в здании)
    'count_shops',          # количество магазинов в здании
    'count_top10',          # количество самых частых магазинов в зданиях (самые частые магазины в jupyter notebook)
    'count_building_200',   # количество зданий в радиусе 200 метров
    'count_shop_mall_200',  # количество торговых центров в радиусе 200 метров
    'area_200_sum',         # суммарная площадь зданий в радиусе 200 метров
    'area_200_max',         # максимальная площадь здания в радиусе 200 метров
    'area_200_mean',        # средняя площадь зданий в радиусе 200 метров
    'count_shops_200',      # количество магазинов в зданиях в радиусе 200 метров
    'count_top10_200',      # количество самых частых магазинов в зданиях в радиусе 200 метров
    'mid',
    'lat',
    'lon',
]

train_ids = set(train_data.mid)

count_all_ids = transactions.mid.unique().shape[0]

train_array = np.zeros((200000, len(columns)))
train_index = 0
test_array = np.zeros((200000, len(columns)))
test_index = 0

buildings = buildings_lib.Buildings()


for _, row in tqdm(transactions.iterrows(), total=transactions.shape[0]):
    mid, lat, lon = row.mid, row.lat, row.lon

    if mid in train_ids:
        array = train_array
        index = train_index
        train_index += 1
    else:
        array = test_array
        index = test_index
        test_index += 1

    bs = buildings.get_buildings_in_circle_any(lat, lon, 200, 200)
    if len(bs) > 0:
        for b in bs:
            if b.is_point_in_building(lat, lon):
                array[index, 0] = 1
                array[index, 1] = int(b.is_shop_mall_or_supermarket())
                array[index, 2] = int(b.get_area())
                array[index, 3] = len(b.shops)
                array[index, 4] = b.count_good_names()
                break

        array[index, 5] = len(bs)
        array[index, 6] = sum([b.is_shop_mall_or_supermarket() for b in bs])
        areas = [b.get_area() for b in bs]
        array[index, 7] = sum(areas)
        array[index, 8] = max(areas)
        array[index, 9] = array[index, 7] / len(bs)
        array[index, 10] = sum([len(b.shops) for b in bs])
        array[index, 11] = sum([b.count_good_names() for b in bs])

    array[index, 12] = mid
    array[index, 13] = lat
    array[index, 14] = lon


def write_to_csv(np_array, real_size, csv_name):
    df = pd.DataFrame(np_array[:real_size, :])
    df.columns = columns
    df.to_csv('features/features_buildings_{}.csv'.format(csv_name), index=False)

write_to_csv(train_array, train_index, 'train')
write_to_csv(test_array, test_index, 'test')
