import argparse
from tqdm import tqdm
import simplejson as json
import yaml
import torch
from torch import nn
import torch.nn.functional as F

from pathlib import Path
from torchnet.meter import AverageValueMeter
from .dataset.statedataset import StateDataset
from .models.state_prediction import StatePredictionModel
from .dataset.lmdb import LMDBDataset
from torch.utils.data import DataLoader
from .utils.spatial_cross_entropy import spatial_cross_entropy

from .datasets_cfg import DS_STATS

from torch.backends import cudnn
cudnn.benchmark = True


def parse_args(*args, **kwargs):
    parser = argparse.ArgumentParser('Training the visual state prediction')

    parser.add_argument('--save', default='', type=str)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--resume-config', action='store_true')

    # Data
    # Dataloading
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--epoch_size', default=900, type=int, help='number of batch per epoch')
    parser.add_argument('--epochs', default=600, type=int, help='number of epochs')
    parser.add_argument('-j', '--workers', default=20, type=int, metavar='N',
                        help='number of workers in dataloader')

    # Dataset path
    parser.add_argument('--ds_root', type=str, default='dataset/synthetic-shapes-1to6')
    parser.add_argument('--lmdb_readahead', action='store_true')

    # Loss
    parser.add_argument('--field-alpha', default=1, type=float)
    parser.add_argument('--mask-alpha', default=1, type=float)
    parser.add_argument('--class-alpha', default=1, type=float)
    parser.add_argument('--depth-alpha', default=1, type=float)

    # Optimizer
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--weight_decay', default=0.0000, type=float)
    parser.add_argument('--lr_epoch_decay', default=250, type=int)

    return parser.parse_args(*args, **kwargs)


def adjust_learning_rate(optimizer: torch.optim.Optimizer,
                         epoch: int, lr_init: float, lr_epoch_decay: int):
    """
    Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = lr_init * (0.1 ** (epoch // lr_epoch_decay))
    print(f'Epoch {epoch}: using lr={lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main(ext_args=None):
    if ext_args is not None:
        args = parse_args([])
        vars(args).update(ext_args)
    else:
        args = parse_args()

    if args.resume_config:
        resume_args = yaml.load((Path(args.resume).parent / 'config.yaml').read_text())
        vars(args).update({k: v for k, v in vars(resume_args).items() if 'resume' not in k})

    print(args)
    have_cuda = torch.cuda.is_available()

    def cast(obj):
        return obj.cuda() if have_cuda else obj

    epoch_size = args.epoch_size * args.batch_size
    scene_kwargs = dict(db_dir=args.ds_root, readahead=args.lmdb_readahead)
    scene_ds_train = LMDBDataset(train=True, **scene_kwargs)
    scene_ds_val = LMDBDataset(train=False, **scene_kwargs)
    ds_kwargs = dict(input_resize=(240, 320), coord_in_base_frame=True, field_size=(69, 85))
    ds_train = StateDataset(scene_ds_train, epoch_size=epoch_size, **ds_kwargs)
    ds_val = StateDataset(scene_ds_val, epoch_size=int(epoch_size * 0.1), **ds_kwargs)

    model = nn.DataParallel(StatePredictionModel())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    start_epoch = 0
    if args.resume:
        path = Path(args.resume)
        print(f'Loading checkpoing from {path}')
        params = torch.load(path)
        state_dict = params['state_dict']
        model.load_state_dict(state_dict)
        start_epoch = params['epoch'] + 1

    model = cast(model)

    def log(loss_dict, epoch):
        save = Path(args.save)
        loss_dict.update(epoch=epoch)
        save.mkdir(exist_ok=True)
        if not (save / 'config.yaml').exists():
            (save / 'config.yaml').write_text(yaml.dump(args))
        path = save / 'checkpoint.pth.tar'
        torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, path)
        with open(save / 'log.txt', 'a') as f:
            f.write(json.dumps(loss_dict, ignore_nan=True) + '\n')
        print(loss_dict)
        print(save.name)

    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args.lr, args.lr_epoch_decay)
        train_fields = ('total', 'field', 'mask', 'class', 'depth')
        meters_train = {k: AverageValueMeter() for k in train_fields}
        meters_val = {k: AverageValueMeter() for k in train_fields}

        def h(net, data, train):
            inputs = cast(data.inputs)

            field_pred, mask_pred, class_pred, depth_pred = model(inputs)

            # prepare targets
            mean, std = cast(torch.tensor(DS_STATS.xy_mean).float()), cast(torch.tensor(DS_STATS.xy_std).float())
            y = cast(data.field)
            y = (y - mean.view(-1, 1, 1))/(std.view(-1, 1, 1))

            target_mask = cast(data.target_mask)
            mask_cond = target_mask != 0
            extract_cond_pixels = lambda field: field.permute(0, 2, 3, 1)[mask_cond, :]

            l_field = F.mse_loss(extract_cond_pixels(field_pred), extract_cond_pixels(y))

            l_mask = F.binary_cross_entropy_with_logits(mask_pred.squeeze(1),
                                                        target_mask.float(),
                                                        pos_weight=torch.tensor(DS_STATS.mask_pos_weight))

            class_mask_targets = cast(data.target_sem_segm)
            l_class = spatial_cross_entropy(class_pred, class_mask_targets.unsqueeze(1).long())

            depth = cast(data.depth)
            l_depth = F.mse_loss(depth_pred, (depth - DS_STATS.depth_mean) / DS_STATS.depth_std)

            # logging
            meters = meters_train if train else meters_val
            for name, l in zip(('field', 'mask', 'class', 'depth'),
                               (l_field, l_mask, l_class, l_depth)):
                meters[name].add(l.item())
            return args.field_alpha * l_field + args.mask_alpha * l_mask \
                + args.class_alpha * l_class + args.depth_alpha * l_depth

        model.train()
        ds_iter = DataLoader(ds_train, sampler=ds_train.make_sampler(), batch_size=args.batch_size,
                             num_workers=args.workers, collate_fn=ds_train.collate_fn)
        iterator = tqdm(ds_iter, ncols=80)
        for sample in iterator:
            optimizer.zero_grad()
            loss = h(model, sample, train=True)
            iterator.set_postfix(loss=loss.item())
            meters_train['total'].add(loss.item())
            loss.backward()
            optimizer.step()

        model.eval()
        ds_iter = DataLoader(ds_val, sampler=ds_val.make_sampler(), batch_size=args.batch_size,
                             num_workers=args.workers, collate_fn=ds_val.collate_fn)
        with torch.no_grad():
            for sample in tqdm(ds_iter, ncols=80):
                loss = h(model, sample, train=False)
                meters_val['total'].add(loss.item())

        log_dict = dict()
        log_dict.update({
            'train_loss': meters_train['total'].mean,
            'train_loss_field': meters_train['field'].mean,
            'train_loss_mask': meters_train['mask'].mean,
            'train_loss_class': meters_train['class'].mean,
            'train_loss_depth': meters_train['depth'].mean,
            'val_loss': meters_val['total'].mean,
            'val_loss_field': meters_train['field'].mean,
            'val_loss_mask': meters_val['mask'].mean,
            'val_loss_class': meters_val['class'].mean,
            'val_loss_depth': meters_val['depth'].mean})

        log(log_dict, epoch=epoch)


if __name__ == '__main__':
    main()
