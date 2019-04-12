import warnings
import torch
from torch import nn
from torchvision.models.alexnet import AlexNet, alexnet
import torch.nn.functional as F


class ImageNetExtractor(nn.Module):

    def __init__(self):
        super().__init__()
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, inputs, normalize=False, size=None):
        outputs = self.model((inputs - self.mean) / self.std)
        if normalize:
            outputs = F.normalize(outputs)
        if size is not None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                outputs = F.interpolate(outputs, size=size, mode='bilinear')
        return outputs


class AlexNetFeatures(AlexNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return self.features(x)


class AlexNetFeatureExtractor(ImageNetExtractor):

    def __init__(self):
        super(AlexNetFeatureExtractor, self).__init__()
        self.model = AlexNetFeatures(1000)

        model_full = alexnet(pretrained=True)
        self.model.load_state_dict(model_full.state_dict())
        self.model.features = self.model.features[:9]
