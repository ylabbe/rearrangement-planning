from collections import namedtuple

DS_DIR = './data/datasets/'


REAL_DATASETS = [
    'real-cubes-1to12',
    'real-cubes-1to6',
]
SYNTHETIC_DATASETS = [
    'synthetic-shapes-1to6',
]

DatasetStatistics = namedtuple('DatasetStatistics',
                               ['xy_mean', 'xy_std', 'mask_pos_weight', 'depth_mean', 'depth_std'])

DS_STATS = DatasetStatistics(xy_mean=[-0.4236, 0.13],
                             xy_std=[0.1816, 0.2594],
                             mask_pos_weight=53,
                             depth_mean=0.7514,
                             depth_std=0.2569)
