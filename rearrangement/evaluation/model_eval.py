from pathlib import Path
import numpy as np

from ..dataset.lmdb import LMDBDataset
from ..dataset.real_scene_dataset import RealSceneDataset
from ..datasets_cfg import SYNTHETIC_DATASETS, REAL_DATASETS, DS_DIR


class ModelEvaluation:
    def __init__(self, dataset, config):
        self.dataset = dataset
        self.config = config

        if self.dataset in REAL_DATASETS:
            if dataset == 'real-cubes-1to6':
                self.dataset, n_objects = 'real-cubes-1to12', np.arange(1,7)
            else:
                n_objects = None
            scene_ds = RealSceneDataset(Path(DS_DIR) / self.dataset, n_objects=n_objects)
        elif self.dataset in SYNTHETIC_DATASETS:
            n_eval_frames = 300
            scene_ds = LMDBDataset(db_dir=str(Path(DS_DIR) / self.dataset),
                                   train=False, n_frames=n_eval_frames)
        else:
            raise ValueError('unknown dataset')
        self.scene_ds = scene_ds
