from torch import nn
import torch.nn.functional as F

from .wide_resnet import BasicBlockV2, WideResNet

CONFIG = {18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}


class UpSamplingHead(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(UpSamplingHead, self).__init__()
        self.deconv0_bn = nn.BatchNorm2d(num_inputs)
        self.deconv0 = nn.ConvTranspose2d(num_inputs, 128, kernel_size=3, padding=1,
                                          dilation=2, stride=2, bias=False)
        self.deconv1_bn = nn.BatchNorm2d(128)
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1,
                                          dilation=2, stride=2, bias=False)
        self.deconv2_bn = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, padding=1,
                                          stride=2, bias=False)
        self.deconv3_bn = nn.BatchNorm2d(32)
        self.deconv4 = nn.Conv2d(32, num_outputs, kernel_size=3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data *= 0.001

    def forward(self, x):
        x = F.leaky_relu(self.deconv0_bn(x), inplace=True)
        x = self.deconv0(x)
        x = F.leaky_relu(self.deconv1_bn(x), inplace=True)
        x = self.deconv1(x)
        x = F.leaky_relu(self.deconv2_bn(x), inplace=True)
        x = self.deconv2(x)
        x = F.leaky_relu(self.deconv3_bn(x), inplace=True)
        x = self.deconv4(x)
        return x


class StatePredictionModel(nn.Module):
    def __init__(self, n_inputs=9, n_field_dims=2, n_classes=6):
        super(StatePredictionModel, self).__init__()

        block = BasicBlockV2
        depth, width = 18, 1.0
        self.base = WideResNet(block, CONFIG[depth], width=width, n_inputs=n_inputs)
        n_maps = int(512 * width)
        self.field_head = UpSamplingHead(n_maps, n_field_dims)
        self.mask_head = UpSamplingHead(n_maps, 1)
        self.classification_head = UpSamplingHead(n_maps, n_classes)
        self.depth_head = UpSamplingHead(n_maps, 1)

    def forward(self, renders):
        x = self.base(renders)

        field = self.field_head(x)
        mask = self.mask_head(x)
        class_mask = self.classification_head(x)
        depth = self.depth_head(x)
        return field, mask, class_mask, depth
