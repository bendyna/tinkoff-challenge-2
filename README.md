# tinkoff-challenge-2

https://boosters.pro/champ_3

Чтобы заново сгенерировать некоторые файлы, нужно скачать http://download.geofabrik.de/russia-latest.osm.pbf

Также нужно распаковать osm_geocoder.tar.gz

В jupyter notebook описаны некоторые идеи, есть отрисовка карт для мерчантов и описан общий ход решения.

В файлах features_*.py генерация признаков.

features_combine.py объединяет все признаки в один датасет.

learn.py - обучение xgboost и дополнительные эвристики.
