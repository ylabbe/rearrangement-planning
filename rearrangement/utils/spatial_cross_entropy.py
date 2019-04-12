import torch.nn.functional as F


def spatial_cross_entropy(inputs, targets):
    """Spatial Cross Entropy

    Expects inputs and targets in NCHW format
      * inputs [NCHW] float
      * tagets [N1HW] long
    """
    assert inputs.dim() == targets.dim() == 4
    assert targets.shape[1] == 1
    assert all(inputs.shape[i] == targets.shape[i] for i in (0, 2, 3))
    x = inputs.permute(0, 2, 3, 1).flatten(0, 2)
    y = targets.permute(0, 2, 3, 1).flatten()
    return F.cross_entropy(x, y)
