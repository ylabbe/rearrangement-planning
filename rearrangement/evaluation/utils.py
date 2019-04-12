import torch
import numpy as np
import torch.nn.functional as F

GRIP_WIDTH = 0.085  # Distance between opposite fingers
GRIP_FINGER_WIDTH = 0.031 * 0.8  # Width of one finger


def overlapping_1d(box1, box2):
    xmin1, xmax1 = box1
    xmin2, xmax2 = box2
    return xmax1 >= xmin2 and xmax2 >= xmin1


def strictly_within_1d(box1, box2):
    "box1 within box2"
    xmin1, xmax1 = box1
    xmin2, xmax2 = box2
    return xmin1 > xmin2 and xmax1 < xmax2


def is_grasp_success(aabb, pose, finger_width=GRIP_FINGER_WIDTH, gripper_width=GRIP_WIDTH):
    pose = np.asarray(pose)
    aabb = np.asarray(aabb)
    overlap_one_finger = overlapping_1d(pose[0] + np.asarray([-finger_width, + finger_width]), aabb[:, 0])
    within_gripper_width = strictly_within_1d(aabb[:, 1], pose[1] + np.asarray([-gripper_width / 2, gripper_width / 2]))
    return overlap_one_finger and within_gripper_width


def extract_patches(im, inst_segm, patch_size=(224, 224), n_add_pixels=3):
    assert inst_segm.dim() == 2
    im_size = im.shape[-2:]
    inst_segm = F.interpolate(inst_segm.unsqueeze(0).unsqueeze(0).float(), size=im_size)[0,0]
    uniqs = torch.unique(inst_segm[inst_segm > 0], sorted=True)
    n_objects = len(uniqs)
    patches = torch.zeros(n_objects, 3, *patch_size)
    for k, uniq in enumerate(uniqs):
        ids = (inst_segm == uniq).nonzero()
        topleft, _ = ids.min(dim=0)
        bottomright, _ = ids.max(dim=0)
        topleft = torch.max(topleft - n_add_pixels, torch.tensor([0, 0]))
        bottomright = torch.min(bottomright + n_add_pixels, torch.tensor(im_size) - 1)
        patch = im[:, topleft[0]: bottomright[0], topleft[1]: bottomright[1]]
        patch_resized = F.interpolate(patch.unsqueeze(0).float(), size=patch_size)
        patches[k] = patch_resized[0]
    return patches


def remap_inst_segm(src_inst_segm, tgt_inst_segm, matching):
    src_inst_segm_remapped = torch.zeros_like(src_inst_segm)
    for k, matched_uniq in enumerate(matching.tolist()):
        src_inst_segm_remapped[src_inst_segm == matched_uniq + 1] = (k + 1)
    return src_inst_segm_remapped


def is_arange(uniqs):
    return torch.all(uniqs == torch.arange(len(uniqs)) + 1)


def extract_point_cloud(pose, mask):
    assert mask.dim() == 3
    assert pose.dim() == 4
    assert len(pose) == len(mask)
    objects_pose = []
    for pose_n, mask_n in zip(pose, mask):
        uniqs = torch.unique(mask_n[mask_n > 0], sorted=True).long()
        # assert is_arange(uniqs.cpu()), 'mask is not an instance segmentation mask'
        objects_pose_ = []
        for i in uniqs.tolist():
            mask_cond = mask_n == i
            if torch.any(mask_cond):
                objects_pose_.append(pose_n[:, mask_cond])
        objects_pose.append(objects_pose_)
    return objects_pose


def coord_matching(src_coord, tgt_coord):
    assert src_coord.dim() <= 2
    assert tgt_coord.dim() <= 2
    if src_coord.dim() == 2 and tgt_coord.dim() == 2:
        N, S_D = src_coord.shape
        M, T_D = tgt_coord.shape
        assert S_D == T_D, 'Source and target coord have different dimensions'
        success_matching = M == N and N > 0
    else:
        success_matching = False
    if success_matching:
        if N == 1:
            matching = torch.tensor([0])
        else:
            # Only match one object to one other
            dists = torch.norm(src_coord.unsqueeze(1) - tgt_coord.unsqueeze(0), p=2, dim=-1)
            _, permut = dists.min(-1)[0].sort()
            matching = - torch.ones(N, dtype=torch.long)
            for i, dists_i in enumerate(dists[permut]):
                for j in dists_i.sort()[1]:
                    if j not in matching:
                        matching[permut[i]] = j
                        break
    else:
        matching = None
    return success_matching, matching
