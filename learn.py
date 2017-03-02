#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
from math import sqrt
import numpy as np
import xgboost as xgb
from tqdm import tqdm
from random import shuffle, seed
from collections import defaultdict

seed(113)
HSIDE = 0.002


def get_accuracy_score(arr):
    correct = 0
    for i in xrange(arr.shape[0]):
        if abs(arr[i, 1] - train_ll[arr[i, 0]][0]) < HSIDE and abs(arr[i, 2] - train_ll[arr[i, 0]][1]) < HSIDE:
            correct += 1
    return correct * 1.0 / arr.shape[0]


# Функция, которая находит оптимальную точку среди транзакций и сдвигает её, в сторону известных мерчантов
# trans_prob    - датафрейм с полями mid, lat, lon, prob
# train_mids    - set merchant_id, которые участвуют в обучении
# bias_factor   - можно сдвигать в сторону мерчантов, которые находятся на расстоянии
#                 не больше bias_factor * 0.002 по широте и долготе
# merge_factor  - можно объединять группу транзакций, которые находятся на расстоянии не больше
#                 merge_factor * 0.002 по широте и долготе
def get_answer_bias(trans_prob, train_mids, bias_factor=1.25, merge_factor=1.2):
    test_mids = set(trans_prob.mid)
    res = np.zeros((len(test_mids), 3))
    for i, mid in enumerate(trans_prob.mid.unique()):
        res[i, 0] = mid
        trans = trans_prob[trans_prob.mid == mid]

        trans_max = trans[trans.prob == trans.prob.max()].iloc[0, :]
        maxlat, maxlon = trans_max.lat, trans_max.lon
        # несколько мерчантов, которые мы подправляем вручную (смотри jupyter notebook)
        # формат ((top, right, bottom, left), (fix_lat, fix_lon))
        # если предсказанный мерчант попадает в прямоугольник [bottom, top]x[left, right],
        # то заменяем на (fix_lat, fix_lon)
        fix_rects = [
            ((59.895223, 30.518269, 59.889961, 30.509643), (59.891647, 30.522085)),  # Мега Дыбенко
            ((55.606780, 37.496472, 55.602140, 37.488480), (55.605617, 37.488462)),  # Мега Теплый Стан
            ((55.914003, 37.402805, 55.908869, 37.393878), (55.911360, 37.397)),     # Мега Химки
        ]
        fixed = False
        for rect in fix_rects:
            if rect[0][0] > maxlat > rect[0][2] and rect[0][1] > maxlon > rect[0][3]:
                res[i, 1:3] = rect[1]
                fixed = True
                break
        if fixed:
            continue

        # Оставляем только уникальные геоточки (среди одинаковых выбираем с наибольшей вероятностью)
        ll_prob = {}
        for j, row in trans.iterrows():
            ll_key = (row.lat, row.lon)
            if ll_key not in ll_prob or row.prob > ll_prob[ll_key]:
                ll_prob[ll_key] = row.prob

        ll_prob = [(ll_key[0], ll_key[1], v) for ll_key, v in ll_prob.iteritems()]

        # Фильтруем слишком маленькие вероятности
        ll_prob_filtered = [ll for ll in ll_prob if ll[2] > 0.5]
        if len(ll_prob_filtered) > 0:
            ll_prob = ll_prob_filtered

        # Далее находим группу транзакций с максимальной суммой вероятностей,
        # при этом, чтобы каждая пара из них отстояла друг от друга по широте и долготе
        # не более, чем на 0.002 * merge_factor
        ll_prob = sorted(ll_prob, key=lambda t: t[0])
        size = len(ll_prob)
        maxp = None
        bottom = -1
        top = 0
        # Для каждой транзакции рассмотрим вариант, что она имеет наименьшую широту в искомой группе транзакций
        while bottom < size - 1:
            bottom += 1
            # Все транзакции в ll_prob отсортированы по широте, находим максимальный индекс, который отстоит
            # не более чем на 0.002 * merge_factor по широте
            while top < size and ll_prob[top][0] < ll_prob[bottom][0] + merge_factor * HSIDE:
                top += 1
            # Фильтруем по широте и долготе
            ll_prob_filtered = [ll for ll in ll_prob[bottom:top] if ll_prob[bottom][1] - merge_factor * HSIDE
                                < ll[1] < ll_prob[bottom][1] + merge_factor * HSIDE]
            ll_prob_filtered = sorted(ll_prob_filtered, key=lambda t: t[1])
            left = -1
            right = 0
            # Из отфильтрованных для каждой рассматриваем вариант, что эта транзакция имеет наименьшую долготу
            while left < len(ll_prob_filtered) - 1:
                left += 1
                # Все транзакции в ll_prob_filtered отсортированы по долготе, находим максимальный индекс,
                # который отстоит не более чем на 0.002 * merge_factor по долготе
                while right < len(ll_prob_filtered) \
                        and ll_prob_filtered[right][1] < ll_prob_filtered[left][1] + merge_factor * HSIDE:
                    right += 1
                p = sum([ll[2] for ll in ll_prob_filtered[left:right]])
                if p > maxp or maxp is None:
                    maxp = p
                    bt = min([ll[0] for ll in ll_prob_filtered[left:right]])
                    tp = max([ll[0] for ll in ll_prob_filtered[left:right]])
                    # Записываем центр ограничивающего прямоугольника группы транзакций
                    res[i, 1] = (bt + tp) / 2
                    res[i, 2] = (ll_prob_filtered[left][1] + ll_prob_filtered[right - 1][1]) / 2

        # Сместим предсказываемую точку, чтобы в прямоугольник попадало как можно больше известных мерчантов
        lat, lon = res[i, 1], res[i, 2]
        known_merchants = defaultdict(int)
        for tm in train_mids:
            mlat, mlon = train_ll[tm]
            if abs(lat - mlat) < HSIDE * bias_factor and abs(lon - mlon) < HSIDE * bias_factor:
                known_merchants[(mlat, mlon)] += 1
        if len(known_merchants) > 0:
            max_sc = 0
            blat = None
            blon = None
            for bottom in known_merchants.keys():
                for left in known_merchants.keys():
                    sc = sum([v for k, v in known_merchants.iteritems() if bottom[0] <= k[0] <= bottom[0] + 2 * HSIDE
                              and left[1] <= k[1] <= left[1] + 2 * HSIDE])
                    if sc > max_sc:
                        max_sc = sc
                        top = max([k[0] for k, v in known_merchants.iteritems()
                                   if bottom[0] <= k[0] <= bottom[0] + 2 * HSIDE
                                   and left[1] <= k[1] <= left[1] + 2 * HSIDE])
                        right = max([k[1] for k, v in known_merchants.iteritems()
                                     if bottom[0] <= k[0] <= bottom[0] + 2 * HSIDE
                                     and left[1] <= k[1] <= left[1] + 2 * HSIDE])
                        ERR = 1e-6  # чтобы избежать ошибок округления и нужные мерчанты точно попали в прямоугольник
                        if lat - HSIDE * bias_factor < bottom[0] < lat - HSIDE:
                            blat = bottom[0] - ERR + HSIDE
                        elif lat + HSIDE * bias_factor > top > lat + HSIDE:
                            blat = top + ERR - HSIDE
                        else:
                            blat = lat
                        if lon - HSIDE * bias_factor < left[1] < lon - HSIDE:
                            blon = left[1] - ERR + HSIDE
                        elif lon + HSIDE * bias_factor > right > lon + HSIDE:
                            blon = right + ERR - HSIDE
                        else:
                            blon = lon
            res[i, 1] = blat
            res[i, 2] = blon

    return res


# Изменение признака count_point_all. Так как он зависит от известных транзакций, то для кросс-валидации его надо
# считать по всем трейн транзакциям, а при сабмите по всем трейн+тест транзакциям
def add_features(data):
    ll = data.lat.apply(str) + '_' + data.lon.apply(str)
    vc = ll.value_counts()
    data['count_point_all'] = ll.apply(lambda x: vc[x])


count_merchant_dist = [(d, 'count_merchants_' + str(d)[2:]) for d in [0.001, 0.002, 0.003, 0.004, 0.006, 0.01]]


# Добавляет признаки по известным трейн мерчантам, поэтому эту функцию надо вызывать снова при каждом разбиении
# на трейн и кросс-валидацию
# Добавляет для транзакции минимальное расстояние до мерчанта и количество мерчантов в прямоугольниках
# описываемых в count_merchant_dist
def add_features_mids(data, mids):
    cm = [[] for _ in count_merchant_dist]
    lats = data.lat.values
    lons = data.lon.values
    ms = data.mid.values
    min_ds = []
    for i in xrange(data.shape[0]):
        rlat = lats[i]
        rlon = lons[i]
        rmid = ms[i]
        lat_key = int(rlat * 100)
        lon_key = int(rlon * 100)
        c = [0 for _ in count_merchant_dist]
        min_d = 10000000
        for lt in xrange(lat_key - 1, lat_key + 2):
            for ln in xrange(lon_key - 1, lon_key + 2):
                for mid, lat, lon in train_ll_rects[lt * 100000 + ln]:
                    if mid not in mids or mid == rmid:
                        continue
                    dlat = abs(lat - rlat)
                    dlon = abs(lon - rlon)
                    min_d = min(min_d, sqrt(dlat ** 2 + dlon ** 2))
                    if dlat > 0.01 or dlon > 0.01:
                        continue
                    for j, (d, name) in enumerate(count_merchant_dist):
                        if dlat < d and dlon < d:
                            c[j] += 1
        for j, _ in enumerate(count_merchant_dist):
            cm[j].append(c[j])
        min_ds.append(min_d)
    for j, (_, name) in enumerate(count_merchant_dist):
        data[name] = cm[j]
        data['min_merchant_dist'] = min_ds

train_ll = pd.read_csv('data/mtrain.csv')
train_ll = {row.mid: (row.lat, row.lon) for i, row in train_ll.iterrows()}
# Разбиваем все мерчанты по прямоугольникам размером 0.01 на 0.01 для быстрого поиска соседних мерчантов
train_ll_rects = defaultdict(list)
for merchant_id, latlon in train_ll.iteritems():
    key = int(latlon[0] * 100) * 100000 + int(latlon[1] * 100)
    train_ll_rects[key].append((merchant_id, latlon[0], latlon[1]))

test_data = pd.read_csv('features/features_all_test.csv')
all_train_data = pd.read_csv('features/features_all_train.csv')

for dataset in [all_train_data, test_data]:
    for _, column_name in count_merchant_dist:
        dataset[column_name] = 0
    dataset['min_merchant_dist'] = 0
columns_learn = [c for c in all_train_data.columns if c not in {'target', 'mid', 'lat', 'lon'}]


def get_percentile_uniq(arr):
    arr_uniq = list(set(arr))
    if len(arr_uniq) > 1:
        arr_uniq = sorted(enumerate(arr_uniq), key=lambda x: x[1])
        dict_uniq = {x[1]: k * 1.0 / (len(arr_uniq) - 1) for k, x in enumerate(arr_uniq)}
        res = [dict_uniq[x] for x in arr]
    else:
        res = [0 for _ in arr]
    return res


def add_percentile(data, column):
    perc_column = []
    current_mid = None
    current_values = []
    for _, row in data[['mid', column]].iterrows():
        mid, value = row.mid, row[column]
        if current_mid is not None and mid != current_mid:
            current_values = get_percentile_uniq(current_values)
            perc_column.extend(current_values)
            current_values = []

        current_mid = mid
        current_values.append(value)
    current_values = get_percentile_uniq(current_values)
    perc_column.extend(current_values)
    cn = column + '_percentile'
    data[cn] = perc_column
    if cn not in columns_learn:
        columns_learn.append(cn)

# Добавим перцентили к некоторым признакам
for d in [all_train_data, test_data]:
    add_percentile(d, 'area_200_max')
    add_percentile(d, 'area_200_mean')
    add_percentile(d, 'area_200_sum')
    add_percentile(d, 'count_shops_200')
    add_percentile(d, 'area')

# Параметры xgboost
param = {'max_depth': 8, 'eta': 0.1, 'silent': 1, 'gamma': 0, 'objective': 'binary:logistic',
         'subsample': 0.99, 'colsample_bytree': 0.99, 'scale_pos_weight': 1, 'min_child_weight': 3,
         'alpha': 0, 'seed': 0}
count_trees = 90

SUBMIT = True

if not SUBMIT:
    add_features(all_train_data)
    all_train_ids = list(set(all_train_data.mid))
    cv_count = 10
    cv_len = int(0.3 * len(all_train_ids))
    score = 0
    for _ in tqdm(xrange(cv_count)):
        shuffle(all_train_ids)
        train_ids_set = set(all_train_ids[cv_len:])
        add_features_mids(all_train_data, train_ids_set)
        train_index = all_train_data.mid.apply(lambda x: x in train_ids_set)
        test_index = all_train_data.mid.apply(lambda x: x not in train_ids_set)
        train_data = all_train_data[train_index]
        cv_data = all_train_data[test_index]
        train_target = all_train_data.target[train_index]
        cv_target = all_train_data.target[test_index]

        dtrain = xgb.DMatrix(train_data[columns_learn], label=train_data.target)
        dcv = xgb.DMatrix(cv_data[columns_learn], label=cv_data.target)
        bst = xgb.train(param, dtrain, count_trees)
        cv_predict = bst.predict(dcv)
        answer = pd.concat([all_train_data[['mid', 'lat', 'lon']][test_index],
                            pd.Series(cv_predict, name='prob', index=all_train_data.mid[test_index].index)], axis=1)
        score += get_accuracy_score(get_answer_bias(answer, train_ids_set, bias_factor=1.55))
    print score / cv_count
else:
    add_features(pd.concat([all_train_data, test_data]))
    add_features_mids(all_train_data, set(all_train_data.mid))
    add_features_mids(test_data, set(all_train_data.mid))
    dtrain = xgb.DMatrix(all_train_data[columns_learn], label=all_train_data.target)
    dtest = xgb.DMatrix(test_data[columns_learn])
    bst = xgb.train(param, dtrain, count_trees)
    test_predict = bst.predict(dtest)
    test_answer = pd.concat([test_data[['mid', 'lat', 'lon']], pd.Series(test_predict, name='prob')], axis=1)
    test_answer = get_answer_bias(test_answer, set(all_train_data.mid), bias_factor=1.55)

    # Добавляем фиксированную точку для мерчантов, у которых все транзакции плохие (смотри jupyter notebook)
    answer_ids = set(test_answer[:, 0])
    add_mid = set()
    for merchant_id in pd.read_csv('data/mtest.csv').mid:
        if merchant_id not in answer_ids:
            add_mid.add(merchant_id)

    test_answer_add = np.zeros((len(add_mid), 3))
    test_answer_add[:, 0] = list(add_mid)
    test_answer_add[:, 1] = 55.740213
    test_answer_add[:, 2] = 37.656371
    test_answer = np.vstack((test_answer, test_answer_add))

    np.savetxt('submission.csv', test_answer, delimiter=',', header='_ID_,_LAT_,_LON_', fmt=['%d', '%.5f', '%.5f'],
               comments='')
