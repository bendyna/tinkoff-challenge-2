#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import os
from collections import defaultdict

# Честное вычисление расстояния по координатам точек
# http://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
def haversine(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371000 # Radius of earth in meters
    return c * r

# Честное вычисление работает очень долго, потому что в нем много тригонометрических функций
# Напишем более быструю функцию для нахождения маленьких расстояний (в этом случае поверхность Земли можно
# рассматривать как плоскость)

# длина в метрах одной сотой градуса по широте
LAT_001_DIST = haversine(0, 0, 0.01, 0)
# длина в метрах одной сотой градуса по долготе для каждой широты с шагом 0.01
LON_001_DIST = [haversine(i / 100.0, 0, i / 100.0, 0.01) for i in xrange(7500)]

def haversine_fast_sq(lat1, lon1, lat2, lon2):
    y = (lat1 - lat2) * LAT_001_DIST * 100
    key = int(100.0 * lat1)
    x = (lon1 - lon2) * LON_001_DIST[key] * 100
    return x * x + y * y

def haversine_fast(lat1, lon1, lat2, lon2):
    return math.sqrt(haversine_fast_sq(lat1, lon1, lat2, lon2))

# Переводит latlon координаты в x, y относительно origin
# Предполагается, что latlons находятся в первой четверти относительно origin и расстояния не очень
# большие, то есть поверхность можно рассматривать как плоскость
def latlon_to_xy(latlons, origin):
    olat, olon = origin
    return [(haversine_fast(lat, lon, lat, olon), haversine_fast(lat, lon, olat, lon)) for lat, lon in latlons]

# Переводит x, y координаты в latlon относительно origin
def xy_to_latlon(xy, origin):
    olat, olon = origin
    lon = olon + xy[0] / (LON_001_DIST[int(100.0 * olat)] * 100)
    lat = olat + xy[1] / (LAT_001_DIST * 100)
    return lat, lon


class Shop:
    def __init__(self, lat, lon, tags):
        self.lat = lat
        self.lon = lon
        self.tags = tags


class Building:
    def __init__(self, line):
        tokens = line.strip().split('\t\t')
        self.bid = int(tokens[0])
        tags = tokens[1].split(';')
        self.tags = {tags[2 * i]: tags[2 * i + 1] for i in xrange(len(tags) / 2)}
        self.polygons = []
        for token in tokens[2:]:
            coords = token.split('\t')
            self.polygons.append([[float(t) for t in c.split(',')] for c in coords])
        self.shops = []
        self.centers = []
        self.area = None
        self.bound_rect = None

    def add_shop(self, shop):
        self.shops.append(shop)

    def add_center(self, lat, lon):
        self.centers.append((lat, lon))

    # Выпускаем луч вверх и считаем количество пересеченных сторон
    # если нечетное, то точка внутри
    def is_point_in_polygon(self, lat, lon, polygon):
        count_top = 0
        for i in xrange(len(polygon) - 1):
            slat1 = polygon[i][0]
            slon1 = polygon[i][1]
            slat2 = polygon[i + 1][0]
            slon2 = polygon[i + 1][1]
            if slon1 < lon < slon2 or slon2 < lon < slon1:
                top_lat = slat1 + (slat2 - slat1) * (lon - slon1) / (slon2 - slon1)
                if top_lat > lat:
                    count_top += 1
        return count_top % 2 == 1

    def is_point_in_building(self, lat, lon):
        for polygon in self.polygons:
            if self.is_point_in_polygon(lat, lon, polygon):
                return True
        return False

    def dist_to_building(self, lat, lon):
        m = 1000000000
        for polygon in self.polygons:
            for nlat, nlon in polygon:
                h = haversine_fast(lat, lon, nlat, nlon)
                if h < m:
                    m = h
        return m

    def get_any_point(self):
        return self.polygons[0][0] if len(self.polygons) > 0 and len(self.polygons[0]) > 0 else None

    def get_centers(self):
        return self.centers

    def get_center(self):
        if len(self.centers) == 0:
            print self.bid
            assert False
        if len(self.centers) == 1:
            return self.centers[0]
        res = [0, 0]
        for c in self.centers:
            res[0] += c[0]
            res[1] += c[1]
        return (res[0] / len(self.centers), res[1] / len(self.centers))

    def get_area(self):
        return self.area

    # top lat, right lon, bottom lat, left lon 
    def get_bound_rect(self):
        if self.bound_rect is None:
            top = bottom = right = left = None
            for a in self.polygons:
                for lat, lon in a:
                    if top is None or lat > top:
                        top = lat
                    if bottom is None or lat < bottom:
                        bottom = lat
                    if right is None or lon > right:
                        right = lon
                    if left is None or lon < left:
                        left = lon
            self.bound_rect = (top, right, bottom, left)
        return self.bound_rect

    def is_any_point_in_circle(self, lat, lon, dist_sq):
        for polygon in self.polygons:
            for nlat, nlon in polygon:
                if haversine_fast_sq(lat, lon, nlat, nlon) < dist_sq:
                    return True
        return False

    def is_shop_mall(self):
        return 'shop' in self.tags and self.tags['shop'] == 'mall'

    def count_shops_of_type(self, shop_type):
        return sum(['shop' in shop.tags and shop.tags['shop'] == shop_type for shop in self.shops])

    def count_amenities_of_type(self, amenity_type):
        return sum(['amenity' in shop.tags and shop.tags['amenity'] == amenity_type for shop in self.shops])

    def has_shop_with_name(self, name):
        if 'name' in self.tags and self.tags['name'] == name:
            return True
        for shop in self.shops:
            if 'name' in shop.tags and shop.tags['name'] == name:
                return True
        return False

    def is_shop_mall_or_supermarket(self):
        return 'shop' in self.tags and (self.tags['shop'] == 'mall' or self.tags['shop'] == 'supermarket')

    def count_good_names(self):
        good_names = {'Макдоналдс', 'Дикси', 'KFC', 'Шоколадница', 'Евросеть', 'Связной', 'Теремок', 'МТС'}
        res = 0
        if 'name' in self.tags and self.tags['name'] in good_names:
            res += 1
        for shop in self.shops:
            if 'name' in shop.tags and shop.tags['name'] in good_names:
                res += 1
        return res


class Buildings:
    def __init__(self):
        self.rects = defaultdict(list)
        self.building_ids = {}
        with open('data/close_buildings') as f:
            for line in f:
                b = Building(line)
                if len(b.polygons) and b.bid not in self.building_ids > 0:
                    self.add(b)
                    self.building_ids[b.bid] = b
        if os.path.exists('data/close_shops'):
            with open('data/close_shops') as f:
                for line in f:
                    tokens = line.strip().split('\t\t')
                    bid = int(tokens[0])
                    lat, lon = [float(s) for s in tokens[1].split(',')]
                    tags = tokens[2].split(';')
                    tags = {tags[2 * i]: tags[2 * i + 1] for i in xrange(len(tags) / 2)}
                    self.building_ids[bid].add_shop(Shop(lat, lon, tags))
        if os.path.exists('data/buildings_area_center'):
            with open('data/buildings_area_center') as f:
                for line in f:
                    bid, area, centers = line.strip().split('\t')
                    building = self.building_ids[int(bid)]
                    building.area = float(area)
                    for ll in centers.split(';'):
                        lat, lon = ll.split(',')
                        building.add_center(float(lat), float(lon))

    def get_by_id(self, bid):
        return self.building_ids[bid]

    def get_all_bids(self):
        return self.building_ids.keys()

    def add(self, building):
        if not building.get_any_point():
            return
        lat, lon = building.get_any_point()
        rlat = int(lat * 100)
        rlon = int(lon * 100)
        self.rects[(rlat, rlon)].append(building)

    def get_building(self, lat, lon):
        rlat = int(lat * 100)
        rlon = int(lon * 100)
        lat_add = 1
        lon_add = 2
        for i in xrange(rlat - lat_add, rlat + lat_add + 1):
            for j in xrange(rlon - lon_add, rlon + lon_add + 1):
                if (i, j) in self.rects:
                    for building in self.rects[(i, j)]:
                        if building.is_point_in_building(lat, lon):
                            return building
        return None

    def get_closest_building(self, lat, lon, max_dist, tag):
        rlat = int(lat * 100)
        rlon = int(lon * 100)
        lat_add = 1
        lon_add = 2
        min_dist = max_dist
        min_building = None
        for i in xrange(rlat - lat_add, rlat + lat_add + 1):
            for j in xrange(rlon - lon_add, rlon + lon_add + 1):
                if (i, j) not in self.rects:
                    continue
                for building in self.rects[(i, j)]:
                    if tag is None or tag in building.tags:
                        h = building.dist_to_building(lat, lon)
                        if h < min_dist:
                            min_dist = h
                            min_building = building
        return min_building

    # Возвращает список зданий, любая точка которых на расстоянии <= distance метров от lat, lon
    def get_buildings_in_circle_any(self, lat, lon, distance, distance_mall):
        distance_sq = distance ** 2
        distance_mall_sq = distance_mall ** 2
        rlat = int(lat * 100)
        rlon = int(lon * 100)
        # +300 по долготе потому что здание может быть в одном прямоугольнике по первой точке, но
        # при этом иметь точки в соседних прямоугольниках
        # по широте 1, потому что размер по широте 1100 метров, даже с учетом
        # расстояния 600 метров, надо чтобы здание было 500 метров в длину, таких зданий очень мало
        lat_add = 1
        lon_add = int(math.ceil((distance + 300) / LON_001_DIST[rlat]))
        buildings = []
        for i in xrange(rlat - lat_add, rlat + lat_add + 1):
            for j in xrange(rlon - lon_add, rlon + lon_add + 1):
                if (i, j) in self.rects:
                    for building in self.rects[(i, j)]:
                        if not building.is_shop_mall():
                            if building.is_any_point_in_circle(lat, lon, distance_sq):
                                buildings.append(building)
        lat_add = 1
        lon_add = int(math.ceil((distance_mall + 300) / LON_001_DIST[rlat])) + 1
        for i in xrange(rlat - lat_add, rlat + lat_add + 1):
            for j in xrange(rlon - lon_add, rlon + lon_add + 1):
                if (i, j) in self.rects:
                    for building in self.rects[(i, j)]:
                        if building.is_shop_mall():
                            if building.is_any_point_in_circle(lat, lon, distance_mall_sq):
                                buildings.append(building)
        return buildings

    # Возвращает список зданий, центр которых на расстоянии <= distance метров от lat, lon
    def get_buildings_in_circle_center(self, lat, lon, distance, distance_mall):
        rlat = int(lat * 100)
        rlon = int(lon * 100)
        lat_add = 1
        lon_add = int(math.ceil((distance + 300) / LON_001_DIST[rlat]))
        buildings = []
        for i in xrange(rlat - lat_add, rlat + lat_add + 1):
            for j in xrange(rlon - lon_add, rlon + lon_add + 1):
                if (i, j) in self.rects:
                    for building in self.rects[(i, j)]:
                        if not ('shop' in building.tags and building.tags['shop'] == 'mall'):
                            center = building.get_center()
                            if haversine_fast(center[0], center[1], lat, lon) < distance:
                                buildings.append(building)
        lat_add = 1
        lon_add = int(math.ceil((distance_mall + 300) / LON_001_DIST[rlat])) + 1
        for i in xrange(rlat - lat_add, rlat + lat_add + 1):
            for j in xrange(rlon - lon_add, rlon + lon_add + 1):
                if (i, j) in self.rects:
                    for building in self.rects[(i, j)]:
                        if 'shop' in building.tags and building.tags['shop'] == 'mall':
                            center = building.get_center()
                            if haversine_fast(center[0], center[1], lat, lon) < distance_mall:
                                buildings.append(building)
        return buildings
