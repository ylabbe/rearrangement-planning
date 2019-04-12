import unittest
import pytest
import numpy as np
from pathlib import Path

from rearrangement.datasets_cfg import SYNTHETIC_DATASETS, REAL_DATASETS, DS_DIR
from rearrangement.dataset.lmdb import LMDBDataset
from rearrangement.dataset.statedataset import StateDataset
from rearrangement.dataset.real_scene_dataset import RealSceneDataset

ROOT = Path('rearrangement/test')
SAMPLE_ROOT = Path('rearrangement/test/datasets_samples/')
SAMPLE_ROOT.mkdir(exist_ok=True)


class TestDataset(unittest.TestCase):
    @pytest.mark.train
    def test_synthetic_datasets(self):
        for ds in SYNTHETIC_DATASETS:
            p = Path(DS_DIR) / ds
            scene_ds = LMDBDataset(p)
            ds = StateDataset(scene_ds)
            for _ in range(5):
                ds[next(iter(ds.make_sampler()))]

    @pytest.mark.eval
    def test_real_datasets(self):
        for ds in REAL_DATASETS[:1]:
            p = Path(DS_DIR) / ds
            scene_ds = RealSceneDataset(p)
            ds = StateDataset(scene_ds, epoch_size=len(scene_ds))
            for _ in range(5):
                ds[np.random.randint(len(ds))]
