from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..dataset.statedataset import StateDataset
from ..datasets_cfg import REAL_DATASETS

from .model_eval import ModelEvaluation
from .utils import coord_matching, is_grasp_success


class StateEvaluation(ModelEvaluation):

    def __init__(self, dataset, config):
        super().__init__(dataset, config)

        self.ds = StateDataset(self.scene_ds,
                               coord_in_base_frame=False,
                               field_size=(69, 85),
                               remove_out_of_field=dataset not in REAL_DATASETS,
                               random_crop=False,
                               random_rotation=False)

        self.batch_size = 32
        ds_iter = DataLoader(self.ds, num_workers=12, batch_size=self.batch_size,
                             shuffle=False, collate_fn=self.ds.collate_fn)
        print(f"Loading {dataset} validation data")
        self.datas = tqdm(ds_iter)

        self.config = config

        self._add_ground_truth_fields()

    def _gt_pred_matching(self, coord_gt, coord_pred):
        matching_success, matching = coord_matching(coord_gt, coord_pred)
        if matching_success:
            coord_pred_matched = coord_pred[matching].tolist()
        else:
            coord_pred_matched = [[np.nan] * 2] * len(coord_gt)
        return matching_success, coord_pred_matched

    def _evaluate_clustering(self, field_pred, mask_pred, data, mask_threshold=0.90, bandwidth=0.04):
        objects_pred, _ = self.model.extract_objects(field_pred, mask_pred=mask_pred,
                                                     mask_threshold=mask_threshold, bandwidth=bandwidth)
        for n in range(len(field_pred)):
            matching_success, coord_pred = self._gt_pred_matching(data.coord[n], objects_pred.coord[n])
            self.frame_df['identification_success'].append(matching_success)
            self.object_df['coord_pred'] += coord_pred

    def _add_ground_truth_fields(self):
        self.frame_df = defaultdict(list)
        self.object_df = defaultdict(list)

        # Ground truth
        for data in self.datas:
            for n, obs_n in enumerate(data.obs):
                n_objects = len(data.coord[n])
                frame_id = len(self.frame_df['frame'])
                self.frame_df['frame'].append(frame_id)
                self.frame_df['n_objects'].append(n_objects)

                self.object_df['frame'] += [frame_id] * n_objects
                self.object_df['n_objects'] += [n_objects] * n_objects
                self.object_df['coord_gt'] += data.coord[n].tolist()
                self.object_df['aabbs_gt'] += torch.as_tensor(obs_n['aabbs']).tolist()

        self.gt_fields = list(self.frame_df.keys()) + list(self.object_df.keys())

    def _reset(self, model):
        self.model = model
        self.frame_df = defaultdict(list, self.frame_df)
        self.object_df = defaultdict(list, self.object_df)
        for df in (self.frame_df, self.object_df):
            for k in df.keys():
                if k not in self.gt_fields:
                    df[k] = []

    def evaluate(self,
                 model,
                 mask_threshold=0.90,
                 bandwidth=0.04,
                 print_summary=True):
        self._reset(model)

        for data in tqdm(self.datas):
            images = data.inputs.cuda()
            data = data._replace(
                coord=[torch.tensor(x).float() for x in data.coord])

            joints = torch.stack([torch.tensor(x['joints']) for x in data.obs])

            field_pred, mask_pred, class_pred, depth_pred = model.forward_scaled(images, joints=joints)

            self._evaluate_clustering(field_pred, mask_pred, data, mask_threshold=mask_threshold, bandwidth=bandwidth)

        summary = self.make_summary(print_summary=print_summary)
        return summary, self.frame_df, self.object_df

    def make_summary(self, print_summary=True):
        self.frame_df = pd.DataFrame(self.frame_df)
        self.object_df = pd.DataFrame(self.object_df)
        metrics = dict()

        coord_gt, aabbs_gt, n_objects = self.object_df[['coord_gt', 'aabbs_gt', 'n_objects']].values.T
        coord_gt = np.stack(coord_gt)[..., :2]
        aabbs_gt = np.stack(aabbs_gt)[..., :2]

        def per_n_objects_errors(field):
            coord_pred = self.object_df[field].values
            errors = np.stack(coord_pred) - coord_gt
            errors_norm = np.linalg.norm(errors, axis=1)
            per_nobjects_error_xy = {
                c: np.nanmean(errors[n_objects == c], axis=0).tolist() for c in np.unique(n_objects)}
            per_nobjects_error_norm = {
                c: np.nanmean(errors_norm[n_objects == c], axis=0).tolist() for c in np.unique(n_objects)}
            per_nobjects_error_std = {
                c: np.nanstd(errors_norm[n_objects == c], axis=0).tolist() for c in np.unique(n_objects)}
            return per_nobjects_error_norm, per_nobjects_error_xy, per_nobjects_error_std

        def overall_errors(field):
            coord_pred = self.object_df[field].values
            errors = np.stack(coord_pred) - coord_gt
            errors_norm = np.linalg.norm(errors, axis=1)
            return np.nanmean(errors_norm[n_objects > 0]).item(), np.nanstd(errors_norm[n_objects > 0]).item()

        def below_th_errors(field, thresholds=[0.02, 0.03, 'grasp']):
            coord_pred = self.object_df[field].values
            errors = np.stack(coord_pred) - coord_gt
            errors_norm = np.linalg.norm(errors, axis=1)
            n_succ = {n: np.isfinite(errors_norm[n_objects == n]).sum() for n in np.unique(n_objects)}
            grasp_success = np.array(
                [is_grasp_success(aabb_n, coord_pred_n) for aabb_n, coord_pred_n in zip(aabbs_gt, coord_pred)])
            per_n_objects_below_th = dict()
            below_th = dict()
            for th in thresholds:
                if th == 'grasp':
                    per_n_objects_below_th[th] = np.array(
                        [(grasp_success[n_objects == n]).sum() / N for n, N in n_succ.items()]).tolist()
                    below_th[th] = (grasp_success.sum() / np.array(list(n_succ.values())).sum()).item()
                else:
                    per_n_objects_below_th[th] = np.array(
                        [(errors_norm[n_objects == n] <= th).sum() / N for n, N in n_succ.items()]).tolist()
                    below_th[th] = ((errors_norm <= th).sum() / np.array(list(n_succ.values())).sum()).item()
            return per_n_objects_below_th, below_th

        frame_identification_success = self.frame_df['identification_success'].values
        frame_n_objects = self.frame_df['n_objects'].values
        identification_success_n_objects = {
            n: np.nanmean(frame_identification_success[frame_n_objects == n]).tolist() * 100
            for n in np.unique(frame_n_objects).tolist()}
        identification_success = np.nanmean(frame_identification_success[frame_n_objects > 0]).item() * 100
        errors_mean, errors_std = overall_errors('coord_pred')
        errors_mean_n_objects, _, errors_std_n_objects = per_n_objects_errors('coord_pred')
        below_th_n_objects, below_th = below_th_errors('coord_pred')
        metrics.update(identification_success_n_objects=identification_success_n_objects,
                       identification_success=identification_success,
                       errors_mean=errors_mean,
                       errors_std=errors_std,
                       errors_mean_n_objects=errors_mean_n_objects,
                       errors_std_n_objects=errors_std_n_objects,
                       below_th=below_th,
                       below_th_n_objects=below_th_n_objects)

        if print_summary:
            for k, v in metrics.items():
                print(f'{k} : {v}')

        return metrics
