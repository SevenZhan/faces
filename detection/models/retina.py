from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional
from torchvision.models._utils import IntermediateLayerGetter

from utils import Anchors, RegressionTransform



class Context(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        self.block2_1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.block2_2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2)
        )

        self.block3_1 = nn.Sequential(
            nn.Conv2d(in_channels//2, in_channels//2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels//2),
            nn.ReLU(inplace=True)
        )
        self.block3_2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2)
        )

        self.act = nn.ReLU(inplace=True)

    def forward(self, inputs):

        ouput1 = self.block1(inputs)

        ouput2_ = self.block2_1(inputs)
        ouput2 = self.block2_2(ouput2_)

        ouput3_ = self.block3_1(ouput2_)
        ouput3 = self.block3_2(ouput3_)

        ouputs = torch.cat((ouput1, ouput2, ouput3), dim=1)

        return self.act(ouputs)



class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super().__init__()

        self.inner_blocks = nn.ModuleList()
        self.layer_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            if in_channels == 0:
                raise ValueError("in_channels=0 is currently not supported")
            inner_block_module = nn.Conv2d(in_channels, out_channels, 1)
            layer_block_module = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                Context(out_channels)
            )
            self.inner_blocks.append(inner_block_module)
            self.layer_blocks.append(layer_block_module)

    def get_result_from_inner_blocks(self, x, idx):

        num_blocks = 0
        for m in self.inner_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.inner_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def get_result_from_layer_blocks(self, x, idx):

        num_blocks = 0
        for m in self.layer_blocks:
            num_blocks += 1
        if idx < 0:
            idx += num_blocks
        i = 0
        out = x
        for module in self.layer_blocks:
            if i == idx:
                out = module(x)
            i += 1
        return out

    def forward(self, x):

        # unpack OrderedDict into two lists for easier handling
        names = list(x.keys())
        x = list(x.values())

        last_inner = self.get_result_from_inner_blocks(x[-1], -1)
        results = []
        results.append(self.get_result_from_layer_blocks(last_inner, -1))

        for idx in range(len(x) - 2, -1, -1):
            inner_lateral = self.get_result_from_inner_blocks(x[idx], idx)
            feat_shape = inner_lateral.shape[-2:]
            inner_top_down = functional.interpolate(last_inner, size=feat_shape, mode="nearest")
            last_inner = inner_lateral + inner_top_down
            results.insert(0, self.get_result_from_layer_blocks(last_inner, idx))

        # make it back an OrderedDict
        out = OrderedDict([(k, v) for k, v in zip(names, results)])

        return out



class ClassHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_anchors*2, 1, 1, 0)

    def forward(self, inputs):

        ouputs = self.conv(inputs)
        ouputs = ouputs.permute(0, 2, 3, 1).contiguous()

        return ouputs.view(ouputs.size(0), -1, 2)



class BboxHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_anchors*4, 1, 1, 0)

    def forward(self, inputs):

        ouputs = self.conv(inputs)
        ouputs = ouputs.permute(0, 2, 3, 1).contiguous()

        return ouputs.view(ouputs.size(0), -1, 4)



class LarkHead(nn.Module):
    def __init__(self, in_channels=512, num_anchors=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, num_anchors*10, 1, 1, 0)

    def forward(self, inputs):

        ouputs = self.conv(inputs)
        ouputs = ouputs.permute(0, 2, 3, 1).contiguous()

        return ouputs.view(ouputs.size(0), -1, 10)



class Retina(nn.Module):
    def __init__(self, arch='resnet50', pretrained=True, input_size=[640, 640]):
        super().__init__()

        if 'resnet' in arch:
            from torchvision.models import resnet
            self.body = resnet.__dict__[arch](pretrained)
            return_layers = {'layer2': 0, 'layer3': 1, 'layer4': 2}
            self.body = IntermediateLayerGetter(self.body, return_layers)
            channels = [512, 1024, 2048]
        else: # todo: retina face with mobilenet as body
            from torchvision.models import mobilenet_v2
            self.body = mobilenet_v2(pretrained=True)
            return_layers = {}
            self.body = IntermediateLayerGetter(self.body, return_layers)
            channels = []

        self.fpn = FPN(channels, 256)
        self.classifications = [ClassHead(512, 3) for _ in range(len(return_layers))]
        self.boundingboxes = [BboxHead(512, 3) for _ in range(len(return_layers))]
        self.landmarks = [LarkHead(512, 3) for _ in range(len(return_layers))]
        self.anchors = Anchors()(input_size=input_size)
        self.pts_tfms = RegressionTransform()

    def forward(self, inputs):

        features = self.body(inputs)
        features = self.fpn(features)

        classifications = torch.cat([self.classifications[i](feature) for i, feature in features.items()], dim=1)
        boundingboxes = torch.cat([self.boundingboxes[i](feature) for i, feature in features.items()], dim=1)
        landmarks = torch.cat([self.landmarks[i](feature) for i, feature in features.items()], dim=1)

        if self.training:
            return classifications, boundingboxes, landmarks
        else:
            bboxes, lmarks = self.pts_tfms(self.anchors, boundingboxes, landmarks, inputs)

            return classifications, bboxes, lmarks