import torch
import random
import numpy as np
from collections import namedtuple
import torchvision.transforms.functional as TF
from torchvision import transforms
import PIL

from .utils import make_semantic_segm, rotate_state, remove_out_of_frame_objects, OBJECT_KEYS
from .sampler import ResizedRandomSampler
from torch.utils.data import Dataset


StateData = namedtuple('StateData', ['inputs', 'field', 'target_mask', 'target_sem_segm', 'coord', 'obs', 'depth'])


class StateDataset(Dataset):
    def __init__(self,
                 scene_ds,
                 coord_in_base_frame=False,
                 input_resize=(240, 320),
                 field_size=(69, 85),
                 remove_out_of_field=True,
                 random_crop=True,
                 random_rotation=True,
                 epoch_size=900 * 128):

        self.scene_ds = scene_ds
        self.epoch_size = min(epoch_size, len(scene_ds))
        self.coord_in_base_frame = coord_in_base_frame

        self.field_size = field_size
        self.input_resize = input_resize
        self.remove_out_of_field = remove_out_of_field
        self.random_crop = random_crop
        self.random_rotation = random_rotation

    def _make_xy_field(self, mask, pose):
        pose = torch.as_tensor(pose, dtype=torch.float32)
        field = sum([(mask == (i + 1)).float().repeat(2, 1, 1) * v.view(2, 1, 1)
                     for i, v in enumerate(pose)], torch.zeros(2, *mask.shape[-2:]))
        return field

    def __len__(self):
        return self.epoch_size

    @staticmethod
    def collate_fn(batch):
        data = dict()
        for k in batch[0]._fields:
            v = [getattr(x, k) for x in batch]
            if k not in ('obs', 'coord'):
                v = torch.stack(v)
            data[k] = v
        data = StateData(**data)
        return data

    def make_sampler(self):
        return ResizedRandomSampler(len(self.scene_ds), self.epoch_size)

    def __getitem__(self, index):
        im, instance_segm, obs = self.scene_ds[index]
        assert im.shape[:2] == instance_segm.shape

        depth = im[..., [-1]]
        im = im[..., :3]
        im = TF.to_pil_image(np.asarray(im))
        instance_segm = TF.to_pil_image(np.asarray(instance_segm.unsqueeze(-1)))
        depth = TF.to_pil_image(np.asarray(depth))

        # Augmentation
        if self.random_rotation:
            angle = transforms.RandomRotation.get_params((-5, 5))
            im = TF.rotate(im, angle, resample=PIL.Image.BILINEAR)
            instance_segm = TF.rotate(instance_segm, angle, resample=False)
            depth = TF.rotate(depth, angle, resample=False)

        if self.random_crop:
            crop_size = random.uniform(0.9, 1)
            crop_size = (np.flip(im.size)*crop_size).astype(np.int)
            i, j, h, w = transforms.RandomCrop.get_params(im, output_size=crop_size)
            im = TF.crop(im, i, j, h, w)
            instance_segm = TF.crop(instance_segm, i, j, h, w)
            depth = TF.crop(depth, i, j, h, w)

        # Rescale
        im = TF.resize(im, self.input_resize, interpolation=PIL.Image.BILINEAR)
        im = TF.to_tensor(im)
        target_instance_segm = TF.resize(instance_segm, self.field_size, interpolation=PIL.Image.NEAREST)
        target_instance_segm = torch.as_tensor(np.asarray(target_instance_segm)).squeeze(-1).type(torch.uint8)
        depth = TF.resize(depth, self.field_size, interpolation=PIL.Image.BILINEAR)
        depth = TF.to_tensor(depth)

        joints = (torch.as_tensor(obs['joints']) + 2 * np.pi) / (4 * np.pi)
        im = torch.cat((im.float(), joints.expand(*im.shape[1:], -1).permute(2, 0, 1)), dim=0)

        # Semantic segmentation from instance segmentation + categories
        target_sem_segm = make_semantic_segm(target_instance_segm, obs['categories'])

        # Remove robot from instance segmentation
        not_robot = [i + 1 for i, c in enumerate(obs['categories']) if c in self.scene_ds.object_categories.keys()]
        n_objects = max(not_robot) if len(not_robot) > 0 else 0
        target_instance_segm[target_instance_segm > n_objects] = 0
        for k in obs.keys():
            if k in OBJECT_KEYS:
                obs[k] = obs[k][:n_objects]

        # Remove objects out of frames from obs
        if self.remove_out_of_field:
            target_instance_segm, obs = remove_out_of_frame_objects(target_instance_segm, obs, self.scene_ds.categories)

        # Binary mask
        target_mask = (target_instance_segm > 0).long()

        # xy field
        coord = torch.as_tensor(obs['positions']).view(-1, 3)[:, :2]
        if self.coord_in_base_frame and len(coord) > 0:
            coord = rotate_state(coord, obs['joints'])
        field = self._make_xy_field(target_instance_segm, coord)

        for k, v in obs.items():
            obs[k] = np.asarray(v)
        coord = np.asarray(coord)

        data = StateData(inputs=im, field=field, target_mask=target_mask,
                         target_sem_segm=target_sem_segm, coord=coord, depth=depth,
                         obs=obs)
        return data
