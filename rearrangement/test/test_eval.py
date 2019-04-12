import unittest
import pytest
import numpy.testing as npt
import yaml
from pathlib import Path

from rearrangement.evaluation.state_eval import StateEvaluation
from rearrangement.evaluation.scene_prediction import ScenePrediction

SAVE = False


class TestEvaluation(unittest.TestCase):
    def create_model(self):
        log_dir = Path('./data/models')
        run_id = 'state-prediction-71538'
        self.run_id = run_id
        return ScenePrediction.from_run(log_dir, run_id)

    def compare_ref_outputs(self, ref_path, outputs, datasets, save=False):
        if save:
            Path(ref_path).write_text(yaml.dump({self.run_id: outputs}))

        ref_data = yaml.load(Path(ref_path).read_text())[self.run_id]
        for ds in datasets:
            for metric, metric_value in ref_data[ds].items():
                if isinstance(metric_value, dict):
                    for k, v in metric_value.items():
                        npt.assert_almost_equal(v, outputs[ds][metric][k], decimal=5)
                else:
                    npt.assert_almost_equal(metric_value, outputs[ds][metric], decimal=5)

    @pytest.mark.eval
    def test_eval_real(self):
        model = self.create_model()
        datasets = ('real-cubes-1to12', 'real-cubes-1to6')
        outputs = dict()
        for ds in datasets:
            pose_eval = StateEvaluation(ds, model.config)
            outputs[ds], _,_ = pose_eval.evaluate(model)

        self.compare_ref_outputs('rearrangement/test/test_data_real.yaml', outputs, datasets, save=SAVE)
