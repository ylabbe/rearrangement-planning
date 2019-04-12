import pickle
import warnings
from copy import deepcopy
from io import BytesIO
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

from .scene_dataset import SceneDataset
from .utils import remap_robot_mask_ids, ROBOT_IDS

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class LMDBDataset(SceneDataset):

    def __init__(self, db_dir, train=True, readahead=False, n_frames=None):

        db_dir = Path(db_dir)

        key_path = db_dir / (('train' if train else 'test') + '_keys.pkl')
        with open(key_path, 'rb') as f:
            self.keys = pickle.load(f)
        self.keys = [k for list_keys in self.keys for k in list_keys]

        categories_path = db_dir / 'categories.csv'
        self.categories = dict(pd.read_csv(categories_path, squeeze=True, index_col=0))
        self.config = yaml.load((db_dir / 'config.yaml').read_text())
        self.n_cameras = self.config.n_cameras
        self.n_fixed_steps = self.config.n_fixed_steps

        self.object_categories = deepcopy(self.categories)
        n = len(self.object_categories)

        self.categories.update({n+k: l for k, l in enumerate(ROBOT_IDS.keys())})

        if readahead:
            warnings.warn('readahead is on, dataloading might take minutes before starting')

        lmdb_kwargs = dict(readonly=True, max_readers=1, lock=False, readahead=readahead, meminit=False)
        self.db = lmdb.open(str(db_dir), **lmdb_kwargs)
        self.txn = self.db.begin(write=False)
        self.train = train
        self.n_frames = n_frames

        self.frame_index = self.build_frame_index()
        self.frame_index = self._filter_frame_index(self.frame_index)

    def build_frame_index(self):
        parsed_keys = [k.decode('ascii').split('/') for k in self.keys]
        steps = np.array([int(l[-1]) for l in parsed_keys])
        seeds = np.array([int(l[-3]) for l in parsed_keys])
        cam_ids = [int(l[-2].split('_')[-1]) for l in parsed_keys]
        scene_ids, config_ids = np.divmod(steps + seeds * len(np.unique(steps)), self.n_fixed_steps)
        frame_index = pd.DataFrame({'scene_id': scene_ids, 'config_id': config_ids,
                                    'cam_id': cam_ids, 'view_id': 0})
        if self.n_frames is not None:
            frame_index = frame_index.iloc[:self.n_frames]
        return frame_index

    @staticmethod
    def _deserialize_im(im_buf):
        im = Image.open(BytesIO(im_buf))
        return torch.tensor(np.asarray(im))

    def load(self, scene_id, config_id, view_id, cam_id):
        df = self.frame_index
        cond = (df['scene_id'] == scene_id) & (df['config_id'] == config_id)
        cond &= (df['view_id'] == view_id) & (df['cam_id'] == cam_id)
        frame_id = df[cond].index.item()
        key = self.keys[frame_id]
        buf = self.txn.get(key)

        dic = pickle.loads(buf)
        rgb = self._deserialize_im(dic['rgb'])
        depth = self._deserialize_im(dic['depth']).unsqueeze(-1)
        rgbd = torch.cat((rgb, depth), dim=-1)
        mask = self._deserialize_im(dic['mask'])

        obs = pickle.loads(dic['obs'])
        mask, obs = remap_robot_mask_ids(mask, obs, self.categories)
        obs['categories'] = np.asarray(obs['categories'])
        obs = {k: v for k, v in obs.items() if k not in ('projection_matrix', 'render_options')}
        return rgbd, mask, obs
