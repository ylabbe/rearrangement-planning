import argparse
import zipfile
from pathlib import Path
import gdown

MODEL_ID = '1rIuYO6dOC2QFVQoY2_NNAUNLS6pkT_h9'
EVAL_DS_ID = '17q-dCu9gc8fTdQyudcpibAf0BVpBcQpY'
TRAIN_DS_ID = '19aLnahok-qLcBVrZ9rzoe3gvx6I38Uet'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--train', action='store_true')
    return parser.parse_args()


def gdrive_id_to_url(gdrive_id):
    return f'https://drive.google.com/uc?id={gdrive_id}'


def unpack(p):
    print("Extracting...")
    with zipfile.ZipFile(p, 'r') as zipf:
        zipf.extractall(p.parent)
    p.unlink()


def main():
    args = parse_args()
    data_dir = Path('./data')
    data_dir.mkdir(exist_ok=True)
    model_dir = data_dir / 'models'
    ds_dir = data_dir / 'datasets'

    if args.model:
        model_dir.mkdir(exist_ok=True)
        p = model_dir / 'state-prediction-71538.zip'
        if not p.with_suffix('').exists():
            gdown.download(url=gdrive_id_to_url(MODEL_ID), output=p.as_posix(), quiet=False)
            unpack(p)

    if args.eval:
        ds_dir.mkdir(exist_ok=True)
        p = ds_dir / 'real-cubes-1to12.zip'
        if not p.with_suffix('').exists():
            gdown.download(url=gdrive_id_to_url(EVAL_DS_ID), output=p.as_posix(), quiet=False)
            unpack(p)

    if args.train:
        ds_dir.mkdir(exist_ok=True)
        p = ds_dir / 'synthetic-shapes-1to6.zip'
        if not p.with_suffix('').exists():
            gdown.download(url=gdrive_id_to_url(TRAIN_DS_ID), output=p.as_posix(), quiet=False)
            unpack(p)


if __name__ == '__main__':
    main()
