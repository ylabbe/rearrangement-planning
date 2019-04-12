import numpy as np
import torch
import seaborn as sns
from PIL import Image
from rearrangement.evaluation.utils import coord_matching, GRIP_WIDTH, GRIP_FINGER_WIDTH, is_grasp_success
import matplotlib.pyplot as plt

DPI = 96


def aabbs_from_centers(centers, sizes):
    N = len(centers)
    aabbs = np.zeros((N, 2, 2))
    centers = np.array(centers).reshape(N, 2)
    sizes = np.array(sizes).reshape(N, 1)
    aabbs[:, 0] = centers - sizes / 2
    aabbs[:, 1] = centers + sizes / 2
    return aabbs


def plot_workspace(ax, plot_limits=False):
    ax.grid()
    xmin, xmax = 0.25, 0.75
    ymin, ymax = -0.25, 0.25
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(0.25, 0.75, 0.05))
    ax.set_yticks(np.arange(-0.25, 0.25, 0.05))
    if plot_limits:
        ax.add_artist(plt.Rectangle((0.3, -0.2), 0.4, 0.4, color='black', fill=False,
                                    linestyle='--', linewidth=2.0))
    return ax


def plot_gt_objects(ax, aabbs, plot_center=False,
                    target_box=False, alpha=0.1,
                    collision_radius=None,
                    plot_collision_radius=False,
                    collision_alpha=0.1):
    aabbs = np.array(aabbs)
    for aabb, color in zip(aabbs, sns.color_palette() * 2):
        d = aabb[1] - aabb[0]
        center = (aabb[1] + aabb[0]) / 2
        if target_box:
            alpha = 1.0
        r = plt.Rectangle(aabb[0], d[0], d[1], color=color, alpha=alpha)

        if plot_collision_radius:
            ax.add_artist(plt.Circle(center, collision_radius, color='black', fill=True, alpha=collision_alpha))

        ax.add_artist(r)
        if plot_center:
            ax.scatter(center[0], center[1], color=color, marker='x', s=50*2)


def plot_grasp_positions(ax, coords, colors=None):
    if colors is None:
        colors = sns.color_palette() * 2

    for center_n, color_n in zip(coords, colors):
        center_n = np.array(center_n).tolist()
        ax.plot(center_n[0] + np.array([-GRIP_FINGER_WIDTH / 2, GRIP_FINGER_WIDTH / 2]),
                np.full(2, center_n[1] - GRIP_WIDTH / 2), color=color_n)
        ax.plot(center_n[0] + np.array([-GRIP_FINGER_WIDTH, GRIP_FINGER_WIDTH]),
                np.full(2, center_n[1] + GRIP_WIDTH / 2), color=color_n)
        ax.plot(np.full(2, center_n[0]),
                center_n[1] + np.array([-GRIP_WIDTH / 2, GRIP_WIDTH / 2]), color=color_n)


def plot_centers(ax, centers, color=None, plot_gripper=True):
    colors = sns.color_palette() * 2 if color is None else np.full(len(centers), color)
    for center_n, color in zip(centers, colors):
        ax.scatter(center_n[0], center_n[1], color=color, marker='x', s=150*2)


def plot_pc(ax, pc, color=None):
    colors = sns.color_palette() * 2 if color is None else np.full(len(pc), color)
    for pc_n, color in zip(pc, colors):
        ax.scatter(pc_n[0], pc_n[1], color=color, marker='x', s=10*2)


def resize(im, size=(320, 240)):
    return np.array(Image.fromarray(np.array(im)).resize(size))


def make_rgba_instance_segm(instance_segm, colors):
    rgba = np.zeros((*instance_segm.shape, 4), dtype=np.uint8)
    for uniq, color in zip(np.unique(instance_segm[instance_segm > 0]), colors):
        rgba[instance_segm == uniq, :3] = np.array(color) * 255
        rgba[instance_segm == uniq, -1] = 255
    return rgba


def plot_instance_segm_overlay(im, instance_segm, ax,
                               alpha=0.6, show_id=True,
                               show_box=True, plot_im=True):
    instance_segm = resize(instance_segm, tuple(reversed(im.shape[:2])))
    uniqs = np.unique(instance_segm[instance_segm > 0])
    if plot_im:
        ax.imshow(im)
    ax.imshow(make_rgba_instance_segm(instance_segm, sns.color_palette() * 2), alpha=alpha)

    for n, (color, uniq) in enumerate(zip(sns.color_palette() * 2, uniqs)):
        ids = np.transpose(np.nonzero(instance_segm == uniq))

        if show_box:
            topleft = ids.min(axis=0)
            bottomright = ids.max(axis=0)
            wh = bottomright - topleft
            rect = plt.Rectangle((topleft[1], topleft[0]), wh[1], wh[0],
                                 color=color, fill=False, linewidth=1.0, linestyle='--')
            ax.add_artist(rect)
            string = f'{n + 1}' if show_id else ''
            ax.text(topleft[1], topleft[0] - 4, string, color='black')
    return ax


def plot_src_tgt(src_im, src_aabbs, src_pc, src_coord, src_inst_segm,
                 tgt_im, tgt_aabbs, tgt_pc, tgt_coord, tgt_inst_segm,
                 plot_gripper=True, collision_radius=None,
                 plot_im_matching=False, plot_workspace_matching=False,
                 planned_action=None, axs=None):
    if axs is None:
        n_rows = 2
        n_cols = 3
        ax_size = np.array([300, 300]) * (4/3) / DPI
        f, axs = plt.subplots(n_rows, n_cols, figsize=(ax_size[0] * (n_cols + 1), ax_size[1] * n_rows))

    plot_predictions(axs[0], src_im, src_aabbs, src_coord, src_pc, src_inst_segm,
                     plot_mask=False, plot_class_mask=False, show_id=True)

    plot_predictions(axs[1], tgt_im, tgt_aabbs, tgt_coord, tgt_pc, tgt_inst_segm,
                     plot_mask=False, plot_class_mask=False, show_id=True)

    if collision_radius is not None:
        for coord_n in src_coord:
            axs[0, -1].add_artist(plt.Circle(coord_n, collision_radius, color='black', fill=True, alpha=0.1))

    plot_gt_objects(axs[0, -1], tgt_aabbs, plot_center=False, target_box=True)
    return axs


def plot_src_tgt_patches(src_patches, tgt_patches, matching, axs=None):
    if axs is None:
        n_rows = 2
        n_cols = len(tgt_patches.images)
        ax_size = np.array([100, 100]) * (4/3) / DPI
        f, axs = plt.subplots(n_rows, n_cols, figsize=(ax_size[0] * (n_cols + 1), ax_size[1] * n_rows))

    n_plots = max(len(src_patches.images), len(tgt_patches.images))
    n_plots = min(n_plots, len(axs[0]))

    for n in range(n_plots):
        if n < len(src_patches.images):
            axs[0, n].imshow(src_patches.images[n].permute(1, 2, 0))
            axs[0, n].axis('off')
        if n < len(tgt_patches.images):
            axs[1, n].imshow(tgt_patches.images[matching][n].permute(1, 2, 0))
            axs[1, n].axis('off')
    return axs


def plot_predictions(axs, rgb, aabbs, coord, pc, instance_segm,
                     field=None, mask=None, class_mask=None,
                     plot_field=False, plot_instance_segm=True,
                     plot_mask=False, plot_class_mask=False,
                     show_id=True, alpha=0.6, collision_radius=None, plot_grasp=False):
    m = 0

    aabbs = torch.tensor(aabbs)
    if len(aabbs) > 0:
        centers = (aabbs[:, 1] + aabbs[:, 0]) / 2
        success_matching, matching = coord_matching(coord, centers[..., :2].float())
        aabbs = aabbs[matching] if success_matching else aabbs
        success_grasps = [is_grasp_success(aabb, coord_i) for aabb, coord_i in zip(aabbs, coord)]
    else:
        success_grasps = []
        success_matching = False

    if success_matching:
        errors = torch.norm(centers[matching][..., :2].float() - coord.float(), dim=1)
    else:
        errors = None

    axs[m].imshow(rgb)
    axs[m].axis('off')
    m += 1

    if plot_field:
        axs[m].imshow(field[0])
        m += 1
        axs[m].imshow(field[1])
        m += 1

    if plot_instance_segm:
        plot_instance_segm_overlay(rgb, instance_segm, axs[m],
                                   alpha=0.6, show_id=show_id)
        axs[m].axis('off')
        sem_segm = class_mask.argmax(dim=0).type(torch.uint8)
        sem_segm[sem_segm < 4] = 0
        plot_instance_segm_overlay(rgb, sem_segm, axs[m], alpha=0.6, show_box=False)
        m += 1

    if plot_mask:
        axs[m].imshow(mask[0])
        m += 1

    if plot_class_mask:
        axs[m].imshow(class_mask.argmax(dim=0))
        m += 1

    ax = axs[m]
    avg_error = f'{errors.mean(0).item():.4f}' if errors is not None else 'None'
    max_error = f'{errors.max(0)[0]:.4f}' if errors is not None else 'None'
    ax.set_title(
        f'identification: {success_matching}, avg: {avg_error}, max: {max_error}, grasp {sum(success_grasps)}/{len(coord)}')  # noqa
    color = 'black' if (not success_matching and len(aabbs) > 0) else None
    plot_workspace(ax)
    plot_gt_objects(ax, aabbs)
    # plot_pc(ax, pc, color=color)
    plot_centers(ax, coord, color=color)
    if plot_grasp:
        plot_grasp_positions(ax, coord, colors=[('black' if succ else 'red') for succ in success_grasps])


def plot_batch_predictions(rgb, aabbs, objects, segm, pred, N=2,
                           rand=False, plot_field=False, plot_instance_segm=True,
                           plot_mask=False, plot_class_mask=False, show_id=True, plot_grasp=False,
                           axs=None):
    if axs is None:
        n_rows = min(N, len(rgb))
        n_cols = 2 + (2 if plot_field else 0) + sum([plot_instance_segm, plot_mask, plot_class_mask])
        ax_size = np.array([299, 299]) * (4/3) / DPI
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(ax_size[0] * (n_cols + 1), ax_size[1] * n_rows))
        if n_rows == 1:
            axs = axs[np.newaxis]

    indices = np.random.permutation(range(len(rgb)))[:N] if rand else range(N)
    for k, n in enumerate(indices):
        plot_predictions(axs[k], rgb=rgb[n], aabbs=aabbs[n],
                         coord=objects.coord[n], pc=objects.pc[n],
                         instance_segm=segm.instance[n], field=pred.field[n],
                         mask=pred.mask[n], class_mask=pred.class_mask[n],
                         plot_field=plot_field, plot_instance_segm=plot_instance_segm, plot_mask=plot_mask,
                         plot_class_mask=plot_class_mask, show_id=show_id)
    return axs


def plot_boxes(boxes, conf=None, ax=None, color='red', th=0.5):
    if ax is None:
        ax_size = np.array([299, 299]) * (4/3) / DPI
        _, ax = plt.subplots(1, 1, figsize=(ax_size[0] * 2, ax_size[1] * 2))

    if conf is None:
        conf = np.full(boxes.shape[0], 1.)
    else:
        conf = np.array(conf).copy().clip(0, 1)

    plot_workspace(ax)
    for box, conf in zip(boxes, conf):
        wh = box[[2, 3]] - box[[0, 1]]
        if conf > th:
            ax.add_artist(plt.Rectangle((box[0], box[1]), wh[0], wh[1], color=color, alpha=conf, fill=False))
            ax.text(box[0] + wh[0] / 2, box[1] + 0.005, str(float(conf)), color=color)
    return ax
