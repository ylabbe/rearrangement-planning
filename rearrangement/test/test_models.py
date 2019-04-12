import unittest
import pytest
import torch

from rearrangement.models.state_prediction import StatePredictionModel
from rearrangement.models.wide_resnet import WideResNet, BasicBlockV2


class TestModels(unittest.TestCase):
    @pytest.mark.model
    def test_state_prediction_model(self):
        model = StatePredictionModel()
        model(torch.rand(2, 9, 240, 320))

    @pytest.mark.model
    def test_wide_resnet(self):
        model = WideResNet(BasicBlockV2, [2, 2, 2, 2], 0.5, n_inputs=3)
        model(torch.randn(2, 3, 240, 320))
