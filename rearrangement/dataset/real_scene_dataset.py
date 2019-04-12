import numpy as np
import yaml
import collections
import torch
import json
from PIL import Image
import pandas as pd
from pathlib import Path

from .scene_dataset import SceneDataset


class RealSceneDataset(SceneDataset):
    def __init__(self, root, n_objects=None, resize=(320, 240)):
        self.root = Path(root)
        self.name = self.root.name
        self.config = yaml.load((self.root / 'config.yaml').read_text())

        self.scales = dict()
        for i in range(1,13):
            self.scales[f'cube{i}'] = np.array([0.035, 0.035, 0.0175])

        # One category = one name in this dataset
        self.objects = np.unique(sum([scene['objects'] for scene in self.config['scenes']], []))
        self.categories = {i: o for i, o in enumerate(self.objects)}
        self.text_to_category = {o: i for i, o in self.categories.items()}
        self.object_categories = self.categories

        self.resize = resize

        self.frame_index = self.build_frame_index()
        if n_objects is not None:
            if not isinstance(n_objects, collections.abc.Iterable):
                n_objects = [n_objects]
            self.frame_index = self.frame_index[self.frame_index['n_objects'].isin(n_objects)]

    def build_frame_index(self):
        scenes_dir = self.root / 'scenes'
        scene_ids, config_ids, cam_ids, view_ids, n_objects = [], [], [], [], []
        for d in scenes_dir.iterdir():
            scene_id = int(d.name.split('-')[0])
            config_id = int(d.name.split('-')[1])
            n_cameras = len(list(d.glob('0-kinect*-0-rgb.png')))
            n_views = len(list(d.glob('*-kinect-0-rgb.png')))
            scene_ids += [scene_id] * n_cameras * n_views
            config_ids += [config_id] * n_cameras * n_views
            n_objects += [len(self.config['scenes'][scene_id]['objects'])] * n_cameras * n_views
            cam_ids += list(range(n_cameras)) * n_views
            view_ids += list(np.arange(n_views).repeat(n_cameras))

        frame_index = pd.DataFrame({'scene_id': scene_ids, 'config_id': config_ids,
                                    'cam_id': cam_ids, 'view_id': view_ids,
                                    'n_objects':n_objects})
        return frame_index

    def load(self, scene_id, config_id, view_id, cam_id):
        scene_config_dir = self.root / 'scenes' / f'{scene_id}-{config_id}'
        cam = 'kinect' if cam_id == 0 else 'kinect2'
        rgb = Image.open(scene_config_dir / f'{view_id}-{cam}-0-rgb.png')
        if cam == 'kinect2':
            width, height = rgb.size
            new_width, new_height = 1408, 1056
            left = (width - new_width)/2
            top = (height - new_height)/2
            right = (width + new_width)/2
            bottom = (height + new_height)/2
            rgb = rgb.crop((left, top, right, bottom))
        rgb = rgb.resize(self.resize, resample=Image.BILINEAR)
        rgb = torch.tensor(np.array(rgb))

        obs = json.loads((scene_config_dir / 'infos.json').read_text())
        if 'categories' not in obs:
            obs['categories'] = obs['names']
        positions = obs['positions']
        scales = np.stack([self.scales[name] for name in obs['categories']])
        aabbs = tuple(zip(positions - scales / 2, positions + scales / 2))
        obs.update(aabbs=aabbs, categories_txt=obs['categories'])
        obs.update(categories=[self.text_to_category[c] for c in obs['categories_txt']])
        obs.update(names=obs['categories_txt'])

        segm = torch.zeros(rgb.shape[:2])
        return rgb, segm, obs
