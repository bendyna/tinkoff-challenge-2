import pandas as pd

features = ['features_count', 'features_perc_rect', 'features_target',
            'features_percentile', 'features_time',
            'features_count_point', 'features_transaction', 'features_buildings']
datasets = ['train', 'test']

for dataset in datasets:
    all_data = pd.concat([pd.read_csv('features/{}_{}.csv'.format(f, dataset)) for f in features], axis=1)
    all_data.to_csv('features/features_all_{}.csv'.format(dataset), index=False)
