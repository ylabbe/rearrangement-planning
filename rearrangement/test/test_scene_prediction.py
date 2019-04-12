import unittest
import pytest
import torch
from pathlib import Path

from rearrangement.evaluation.scene_prediction import ScenePrediction


class TestScenePrediction(unittest.TestCase):
    def create_model(self):
        log_dir = Path('./data/models')
        run_id = 'state-prediction-71538'
        return ScenePrediction.from_run(log_dir, run_id)

    @pytest.mark.model
    def test_scene_state_prediction(self):
        model = self.create_model()
        n_images = 8
        rgb = torch.rand(n_images, 448, 448, 3) * 255
        joints = torch.rand(n_images, 6)
        objects, segm, pred = model.predict_objects(rgb, joints=joints)

        self.assertEqual(len(objects.coord), n_images)

    @pytest.mark.model
    def test_matching(self):
        model = self.create_model()
        src_im = torch.rand(1, 448, 448, 3) * 255
        src_joints = torch.rand(6)
        tgt_im = torch.rand(1, 448, 448, 3) * 255
        tgt_joints = torch.rand(6)

        images = torch.cat((src_im, tgt_im), dim=0)
        joints = torch.stack((src_joints, tgt_joints))
        objects, segm, pred = model.predict_objects(images, joints=joints)

        src_segm, tgt_segm = segm.instance

        succ, matching, dst_patches, src_patches, dists = model.match_src_tgt_objects(
            src_im=src_im[0], src_inst_segm=src_segm, tgt_im=tgt_im[0], tgt_inst_segm=tgt_segm)
