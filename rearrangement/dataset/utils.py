import torch
from copy import deepcopy

import numpy as np

ROBOT_IDS = dict(robot=(255, 254, 253, 252, 251, 250, 249),
                 gripper=(248, 247, 246,245,244,243,242,241,242,240,239,
                          238,237,236,235,234,233))

OBJECT_KEYS = ('categories', 'positions', 'names', 'scales', 'categories_txt', 'orientations', 'aabbs')


def rotate_state(state, joints):
    th = joints[0]
    sin_th = np.sin(th)
    cos_th = np.cos(th)
    state = np.asarray(state) @ np.asarray([[cos_th, sin_th], [-sin_th, cos_th]]).T
    return state


def remap_robot_mask_ids(mask, obs, categories_dict):
    mask = np.asarray(mask)
    obs = deepcopy(obs)
    text_to_ids = ROBOT_IDS
    mask_ids_to_text = {v: k for k, v in text_to_ids.items()}
    text_to_category = {v: k for k, v in categories_dict.items() if v in text_to_ids.keys()}
    n_max = len(obs['categories'])
    for n, (k, v) in enumerate(mask_ids_to_text.items()):
        mask[np.isin(mask, k)] = n_max + n + 1
        obs['categories'].append(text_to_category[v])
        obs['categories_txt'].append(v)
    mask = torch.as_tensor(mask)
    return mask, obs


def remove_out_of_frame_objects(mask, obs, categories_dict):
    obs = deepcopy(obs)
    instance_segm = torch.zeros_like(mask)
    filtered_obs = {k: [] for k in OBJECT_KEYS}
    for k, v in obs.items():
        if k in OBJECT_KEYS:
            obs[k] = np.asarray(v).tolist()

    for k, i in enumerate(torch.unique(mask[mask > 0], sorted=True).tolist()):
        for obj_k in OBJECT_KEYS:
            filtered_obs[obj_k].append(obs[obj_k][i-1])
        instance_segm[mask == i] = k + 1

    for k, v in filtered_obs.items():
        if k not in ('categories_txt', 'names'):
            obs[k] = torch.as_tensor(v)
        else:
            obs[k] = v
    return instance_segm, obs


def make_semantic_segm(instance_segm,
                       categories_ids):
    categories_ids = torch.as_tensor(categories_ids, dtype=torch.uint8)
    semantic_segm = torch.zeros_like(instance_segm)
    for k, i in enumerate(torch.unique(instance_segm[instance_segm > 0], sorted=True).tolist()):
        semantic_segm[instance_segm == i] = categories_ids[i-1] + 1
    return semantic_segm
