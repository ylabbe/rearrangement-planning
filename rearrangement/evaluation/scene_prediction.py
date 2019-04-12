from pathlib import Path
from collections import namedtuple
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import MeanShift
import yaml

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .utils import extract_patches, extract_point_cloud

from ..datasets_cfg import DS_STATS

from ..models.state_prediction import StatePredictionModel
from ..models.features_pretrained import AlexNetFeatureExtractor

ObjectPredictions = namedtuple('ObjectPredictions', ['coord', 'pc'])

Segmentation = namedtuple('Fields', ['instance'])

Predictions = namedtuple('Predictions', ['field', 'mask', 'class_mask', 'depth'])

Patches = namedtuple('Patches', ['images', 'descriptors'])


def filter_meanshift_outputs(meanshift, n_pts=2):
    """
    Remove clusters which have less than n_pts points
    """
    labels_ = meanshift.labels_
    centers_ = meanshift.cluster_centers_
    n_pts_per_label = {k: (labels_ == k).sum() for k in np.unique(labels_)}
    labels_filtered = np.ones_like(labels_) * 255
    centers_filtered = []

    k = 0
    for label in np.unique(labels_):
        if n_pts_per_label[label] > n_pts:
            labels_filtered[labels_ == label] = k
            centers_filtered.append(centers_[label])
            k += 1
        else:
            labels_filtered[labels_ == label] = 254
    centers_filtered = torch.as_tensor(centers_filtered).view(-1, 2)
    return torch.as_tensor(labels_filtered), centers_filtered


class ScenePrediction:
    def __init__(self, model, config):
        model.eval()
        self.model = model
        self.config = config
        self.load_alexnet()

    @staticmethod
    def from_run(log_dir, run_id, cuda=True, **kwargs):
        log_dir = Path(log_dir)
        config = yaml.load((log_dir / run_id / 'config.yaml').read_text())
        model = nn.DataParallel(StatePredictionModel())
        checkpoint = torch.load(log_dir / run_id / 'checkpoint.pth.tar')
        if cuda:
            model.cuda()
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        return ScenePrediction(model, config)

    def load_alexnet(self):
        feature_extractor = AlexNetFeatureExtractor().cuda()
        feature_extractor.eval()
        self.alexnet = feature_extractor

    @torch.no_grad()
    def forward_scaled(self, images, joints=None):
        field_pred, mask_pred, class_pred, depth_pred = self.model(images)
        mean = torch.tensor(DS_STATS.xy_mean).float().cuda()
        std = torch.tensor(DS_STATS.xy_std).float().cuda()
        field_pred = (field_pred * std.view(-1, 1, 1)) + mean.view(-1, 1, 1)
        depth_pred = depth_pred * DS_STATS.depth_std + DS_STATS.depth_mean

        #  Convert xy predicted to world frame
        assert joints is not None
        assert joints.shape == (len(field_pred), 6)
        th = joints[:, 0]
        sin_th, cos_th = np.sin(th), np.cos(th)
        R = torch.stack([
            torch.as_tensor([[c_n, s_n],
                             [-s_n, c_n]]).t()
            for c_n, s_n in zip(cos_th.tolist(), sin_th.tolist())]).cuda()
        R = R.unsqueeze(1).unsqueeze(1)
        field_pred = (R @ field_pred.permute(0, 2, 3, 1).unsqueeze(-1)).squeeze(-1).permute(0, 3, 1, 2)
        mask_pred = torch.sigmoid(mask_pred)
        class_pred = F.softmax(class_pred, dim=1)
        return field_pred, mask_pred, class_pred, depth_pred

    def input_transform(self, images, joints=None):
        images = torch.as_tensor(images).cuda()
        assert images.dim() == 4
        assert images.shape[-1] >= 3
        assert images.max() > 1, 'Input must be uint8'

        images = images[..., :3]
        images = images.permute(0, 3, 1, 2).float() / 255.
        if tuple(images.shape[2:]) != (240, 320):
            images = F.interpolate(images, size=(240, 320), mode='bilinear')

        joints = (torch.as_tensor(joints).float().cuda() + 2 * np.pi) / (4 * np.pi)
        joints = joints.unsqueeze(-1).unsqueeze(-1).expand(len(images), 6, *images.shape[-2:])
        images = torch.cat((images.float(), joints), dim=1)
        return images

    def cluster_mean_shift(self, field_pred, mask_pred, mask_threshold, bandwidth=0.04):

        segm = torch.zeros_like(mask_pred, dtype=torch.uint8).squeeze(1).cuda()
        mask = (mask_pred > mask_threshold).squeeze(1)
        centers = []
        for n, (field_pred_n, mask_n) in enumerate(zip(field_pred, mask)):
            if mask_n.sum() > 0:
                pc_n = field_pred_n[:, mask_n].cpu().numpy().T
                meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=False).fit(pc_n)
                labels_, centers_ = filter_meanshift_outputs(meanshift)
                segm[n, mask_n] = torch.as_tensor(labels_, dtype=torch.uint8).cuda() + 1
                centers.append(torch.as_tensor(centers_))
            else:
                centers.append(torch.tensor([]))
        segm[segm == 255] = 0
        return segm, centers

    def extract_objects(self, field_pred, mask_pred, mask_threshold=0.9, bandwidth=0.04):
        segm, centers = self.cluster_mean_shift(field_pred, mask_pred, mask_threshold, bandwidth)
        pc = extract_point_cloud(field_pred.cpu(), segm.cpu())
        xy = [torch.as_tensor(c_n) for c_n in centers]
        segm.squeeze_(1)
        objects = ObjectPredictions(coord=xy, pc=pc)
        segm = Segmentation(instance=segm.cpu())
        return objects, segm

    def predict_objects(self, images, joints=None, mask_threshold=0.9, bandwidth=0.04):

        inputs = self.input_transform(images, joints)

        field_pred, mask_pred, class_pred, depth_pred = self.forward_scaled(inputs, joints=joints)

        objects, segmentation = self.extract_objects(field_pred, mask_pred, mask_threshold, bandwidth)

        predictions = Predictions(field=field_pred.cpu(), mask=mask_pred.cpu(),
                                  class_mask=class_pred.cpu(), depth=depth_pred.cpu())
        return objects, segmentation, predictions

    @torch.no_grad()
    def make_descriptors(self, patches):
        if len(patches) == 0:
            return torch.empty(0)
        assert patches.dim() == 4 and patches.shape[1] == 3
        features = F.normalize(self.alexnet(patches.cuda(), normalize=False).flatten(1).cpu())
        return features

    def match_src_tgt_objects(self, src_im, src_inst_segm, tgt_im=None, tgt_inst_segm=None, tgt_descriptors=None):
        patch_size = (64, 64)
        assert src_im.dim() == 3 and src_im.shape[-1] == 3 and src_im.max() > 1
        src_im = torch.as_tensor(src_im).cuda().float().permute(2, 0, 1) / 255
        src_patches = extract_patches(src_im, src_inst_segm, patch_size=patch_size)
        src_descriptors = self.make_descriptors(src_patches)

        if tgt_descriptors is None:
            assert tgt_im is not None and tgt_inst_segm is not None
            assert tgt_im.dim() == 3 and tgt_im.shape[-1] == 3 and tgt_im.max() > 1
            tgt_im = torch.as_tensor(tgt_im).cuda().float().permute(2, 0, 1) / 255
            tgt_patches = extract_patches(tgt_im, tgt_inst_segm, patch_size=patch_size)
            tgt_descriptors = self.make_descriptors(tgt_patches)

        src_patches = Patches(images=src_patches, descriptors=src_descriptors)
        tgt_patches = Patches(images=tgt_patches, descriptors=tgt_descriptors)

        success_matching = len(src_patches) == len(tgt_patches)
        if success_matching and len(src_patches.images) > 0:
            dists = torch.norm(src_descriptors.unsqueeze(0) - tgt_descriptors.unsqueeze(1), dim=-1)
            matching = linear_sum_assignment(dists)[1]
        else:
            matching = None
            dists = None
        return success_matching, matching, src_patches, tgt_patches, dists
