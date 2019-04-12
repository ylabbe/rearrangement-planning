import pytest
import unittest

from rearrangement.main import main as train_main


class TestTraining(unittest.TestCase):

    @pytest.mark.train
    def test_train(self):
        datasets = ['./data/datasets/synthetic-shapes-1to6']
        for ds in datasets:
            args = dict(batch_size=8, epoch_size=16, epochs=1, lmdb_readahead=False, ds_root=ds, workers=0)
            train_main(ext_args=args)
