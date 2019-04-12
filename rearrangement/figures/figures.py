import numpy as np
from itertools import chain
import seaborn as sns
from pathlib import Path
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable


from ..datasets_cfg import DS_DIR
from ..dataset.real_scene_dataset import RealSceneDataset
from ..evaluation.utils import extract_patches
from ..dataset.lmdb import LMDBDataset
from ..utils.visualization import plot_workspace, plot_gt_objects, resize, \
    plot_instance_segm_overlay, plot_pc, plot_centers, coord_matching, make_rgba_instance_segm

mpl.use('pdf')
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
plt.rc('font', family='serif', serif='Times')
COLUMN_WIDTH = 3.487
HEIGHT = COLUMN_WIDTH / 1.618
WORKSPACE = ((0.3, -0.2), (0.7, 0.2))


def remove_tick_labels(ax):
    for n, label in enumerate(ax.xaxis.get_ticklabels()):
        if n % 2 == 0:
            label.set_visible(False)

    for n, label in enumerate(ax.yaxis.get_ticklabels()):
        if n % 2 == 0:
            label.set_visible(False)


def plot_src_tgt_config(src_aabbs=None,
                        tgt_aabbs=None,
                        radius=0.055,
                        path='figures/configs.pdf'):
    width = COLUMN_WIDTH * 2
    height = COLUMN_WIDTH
    fig, axs = plt.subplots(1, 2, figsize=(width, height))

    for ax, aabbs, title in zip(axs, (src_aabbs, tgt_aabbs), ('Source', 'Target')):
        plot_gt_objects(ax, aabbs, plot_center=False, target_box=False,
                        alpha=1.0, collision_alpha=0.1,
                        collision_radius=radius, plot_collision_radius=True)
        plot_workspace(ax, plot_limits=True)
        ax.set_axisbelow(True)
        ax.set_title(title)
        remove_tick_labels(ax)
        for tick in chain(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
            tick.label.set_fontsize(12)
            tick.label.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(path)
    return fig, axs


def add_colorbar(ax, im, **kwargs):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(im, cax=cax, **kwargs)
    return cb


def save_real_predictions(model,
                          mask_threshold,
                          bandwidth,
                          im_id,
                          n_objects=12,
                          save_dir='figures/'):
    scene_ds = RealSceneDataset(DS_DIR / 'real-cubes-1to12', n_objects=n_objects,
                                resize=(640, 480))
    rgb, mask, obs = scene_ds[im_id]
    rgb = rgb[..., :3].unsqueeze(0)
    joints = torch.tensor(obs['joints']).unsqueeze(0)
    objects, segm, pred = model.predict_objects(rgb, joints=joints, mask_threshold=mask_threshold, bandwidth=bandwidth)

    width = COLUMN_WIDTH * 4 * 2
    height = COLUMN_WIDTH * 4 * 2
    fig, axs = plt.subplots(4, 4, figsize=(width, height))

    axs[0, 0].imshow(rgb[0])

    _resize = lambda im: resize(im, (640, 480))
    im = axs[0, 1].imshow(_resize(pred.field[0, 0]))
    add_colorbar(axs[0, 1], im)
    im = axs[0, 2].imshow(_resize(pred.field[0, 1]))
    add_colorbar(axs[0, 2], im)
    axs[0, 3].imshow(_resize(pred.mask[0, 0]), cmap='gray')

    axs[1, 0].imshow(_resize(pred.class_mask[0, 0]), cmap='gray')
    axs[1, 1].imshow(_resize(pred.class_mask[0, 1]), cmap='gray')
    axs[1, 2].imshow(_resize(pred.class_mask[0, 2]), cmap='gray')
    axs[1, 3].imshow(_resize(pred.class_mask[0, 3]), cmap='gray')
    axs[2, 0].imshow(_resize(pred.class_mask[0, 4]), cmap='gray')
    axs[2, 1].imshow(_resize(pred.class_mask[0, 5]), cmap='gray')

    mask = pred.mask[0, 0]
    mask[mask >= mask_threshold] = 1.0
    mask[mask < mask_threshold] = 0
    axs[2, 2].imshow(_resize(mask.float()), cmap='gray')
    axs[2, 3].imshow(_resize(pred.class_mask[0].argmax(dim=0).float()))
    plot_instance_segm_overlay(rgb[0], segm.instance[0], axs[3, 0], show_box=False, alpha=1.0)

    sem_segm = pred.class_mask[0].argmax(dim=0).type(torch.uint8)
    sem_segm[sem_segm < 4] = 0
    plot_instance_segm_overlay(rgb[0], sem_segm, axs[3, 1], alpha=0.6, show_box=False)
    plot_instance_segm_overlay(rgb[0], segm.instance[0], axs[3, 1], show_box=False, alpha=1.0, plot_im=False)

    if model.config.depth_prediction:
        axs[3, 2].imshow(pred.depth[0, 0], cmap='gray')
    else:
        fig.delaxes(axs[3, 2])

    rgba_instance_segm = make_rgba_instance_segm(np.array(segm.instance[0]), sns.color_palette() * 2)
    axs[3, 3].imshow(rgba_instance_segm[..., :3])
    axs[3, 3].axis('off')

    save_dir = Path(save_dir)
    for ax, name in zip(axs.flatten(), ('rgb', 'x', 'y', 'mask',
                                        'class_background', 'class_triangle', 'class_cylinder', 'class_cube',
                                        'class_robot', 'class_gripper', 'mask_threshold', 'class_argmax',
                                        'instance', 'instance_robot', 'depth', 'inst_segm')):
        ax.axis('off')
        # extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        # fig.savefig(save_dir / f'ex_{name}.png', bbox_inches=extent)
    fig.savefig(save_dir / f'example_predictions.pdf')

    width = COLUMN_WIDTH * 3
    height = COLUMN_WIDTH
    aabbs, coord = torch.tensor(obs['aabbs']), objects.coord[0]
    centers = (aabbs[:, 1] + aabbs[:, 0]) / 2
    success_matching, matching = coord_matching(coord, centers[..., :2].float())
    aabbs = aabbs[matching] if success_matching else aabbs

    fig, axs = plt.subplots(1, 3, figsize=(width, height))
    plot_workspace(axs[0], plot_limits=True)
    remove_tick_labels(axs[0])
    plot_pc(axs[0], objects.pc[0], color='black')
    axs[0].set_axisbelow(True)

    plot_workspace(axs[1], plot_limits=True)
    remove_tick_labels(axs[1])
    plot_pc(axs[1], objects.pc[0])
    plot_centers(axs[1], objects.coord[0])
    axs[1].set_axisbelow(True)
    fig.subplots_adjust(wspace=0.26, hspace=0.0)

    plot_workspace(axs[2], plot_limits=True)
    remove_tick_labels(axs[2])
    plot_centers(axs[2], objects.coord[0])
    axs[2].set_axisbelow(True)
    fig.subplots_adjust(wspace=0.26, hspace=0.0)
    fig.savefig(save_dir / f'example_predictions_workspace.pdf', bbox_inches='tight')

    # Patches
    patches = extract_patches(rgb[0].permute(2, 0, 1).float() / 255, segm.instance[0], patch_size=(64, 64))
    width = COLUMN_WIDTH * 2
    height = COLUMN_WIDTH * 2
    n_rows, n_cols = 3, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    fig.subplots_adjust(wspace=0.2, hspace=0.2)
    for ax, patch, color in zip(axs.flatten(), patches, sns.color_palette()):
        ax.imshow(patch.permute(1, 2, 0))
        ax.add_artist(plt.Rectangle((0, 0), 62, 62, color=color, fill=False, linewidth=6))
        ax.axis('off')
    fig.savefig(save_dir / f'example_patches.pdf', bbox_inches='tight')


def save_train_test_images(synth_ids,
                           real_ids,
                           save_dir='figures'):
    scene_ds = LMDBDataset(Path('../../data/datasets/') / 'synthetic-shapes-1to6')

    width = COLUMN_WIDTH * 5
    height = width * 3 / 4 * 2 / 3
    n_rows, n_cols = 2, 3
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    for ax, synth_id in zip(axs.flatten(), synth_ids):
        rgb = scene_ds[synth_id][0][..., :3]
        ax.imshow(rgb)
        ax.axis('off')
    save_dir = Path(save_dir)
    fig.savefig(save_dir / f'train_images.pdf', bbox_inches='tight')

    scene_ds = RealSceneDataset(Path('../../data/datasets') / 'real-cubes-1to12', resize=(640, 480))

    width = COLUMN_WIDTH * 4 / 3 * 5
    height = width / 4
    n_rows, n_cols = 1, 4
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    fig.subplots_adjust(wspace=0.0, hspace=0.0)
    for ax, real_id in zip(axs.flatten(), real_ids):
        rgb = scene_ds[real_id][0][..., :3]
        ax.imshow(rgb)
        ax.axis('off')
    save_dir = Path(save_dir)
    fig.savefig(save_dir / f'test_n_objects.pdf', bbox_inches='tight')


def plot_vision_eval(summary, path='figures/plot_variable_n_objects.pdf'):

    width = COLUMN_WIDTH * 4
    height = width / 2
    n_rows, n_cols = 1, 2
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(width, height))
    n_objects = np.array(range(1, 13))
    succ = np.array([summary['identification_success_n_objects'][n] for n in n_objects])

    # axs[0].grid()
    for (th, values), color in zip(summary['below_th_n_objects'].items(), sns.color_palette()):
        if not isinstance(th, str):
            label = f'{th * 100} cm'
        else:
            label = 'grasp.'
        axs[0].plot(n_objects, np.array(values) * 100, color=color, label=label, marker='o')
    axs[0].set_xlabel('Number of objects', size=24)
    axs[0].set_title('Error below threshold (%)', size=24)
    axs[0].set_ylim(0., 105)
    axs[0].set_xticks(np.arange(2, 13, 2))
    axs[0].legend(prop=dict(size=20))
    axs[0].tick_params(axis='x', labelsize=24)
    axs[0].tick_params(axis='y', labelsize=24)

    axs[1].plot(n_objects, succ, color='black', marker='o')
    axs[1].set_xlabel('Number of objects', size=24)
    axs[1].set_ylim(0., 105)
    axs[1].set_title('Object identification success (%)', size=24)
    axs[1].set_xticks(np.arange(2, 13, 2))
    axs[1].tick_params(axis='x', labelsize=24)
    axs[1].tick_params(axis='y', labelsize=24)

    fig.savefig(path, bbox_inches='tight')
