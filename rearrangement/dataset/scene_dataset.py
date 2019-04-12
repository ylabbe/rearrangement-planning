import torch
from torch.utils.data import Dataset


class SceneDataset(Dataset):

    def load(self, scene_id, config_id, cam_id, view_id):
        """
        Returns:
        - rgbd
        - segm: instance_segmentation
        - obs: a dict containing additionnal informations about the scene or camera
        """
        raise NotImplementedError

    def build_frame_index(self):
        """
        Returns:
        A dataframe where each row corresponds to an image.
        Each image is uniquely identified by the following fields:
        - scene_id: Unique id of a scene. A scene is defined by the type of objects,
        the number of objects and the textures on the objects/background.
        - config_id: Index for the configuration of a scene. A scene configuration is defined
        by the position of the objects and their orientation.
        - cam_id: Unique index for a camera.
        - view_id: Index for a scene view. Multiple cameras can share the same view_id.
        """
        raise NotImplementedError

    @staticmethod
    def _filter_frame_index(frame_index,
                            n_scenes=None,
                            n_scene_configs=None,
                            n_cameras=None,
                            n_views=None):
        return frame_index

    def collate_fn(self, batch):
        rgb = torch.stack([x[0] for x in batch])
        segm = torch.stack([x[1] for x in batch])
        stacked_obs = {k: [x[-1][k] for x in batch] for k in batch[0][-1].keys()}
        return rgb, segm, stacked_obs

    def __getitem__(self, idx):
        row = self.frame_index.iloc[int(idx)]
        return self.load(row['scene_id'], row['config_id'],
                         row['view_id'], row['cam_id'])

    def __len__(self):
        return len(self.frame_index)
