import argparse

import torch

from .datasets_cfg import REAL_DATASETS
from .evaluation.scene_prediction import ScenePrediction
from .evaluation.state_eval import StateEvaluation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_id', required=True, type=str)
    parser.add_argument('--log_dir', type=str, default='./data/models/')
    parser.add_argument('--dataset', default='real-cubes-1to6', type=str, choices=REAL_DATASETS + ['all'])
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    print(args)
    datasets = REAL_DATASETS if args.dataset == 'all' else [args.dataset]
    model = ScenePrediction.from_run(args.log_dir, args.run_id)
    output_data = {}
    for ds in datasets:
        print(f'Evaluation on: {ds}')
        state_eval = StateEvaluation(ds, model.config)
        output_data[ds], _, _ = state_eval.evaluate(model)


if __name__ == '__main__':
    main()
