import torch
from torch import nn
from torch.nn import functional



########################### OctConv ###########################
class Sampling(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest'):
        super(Sampling, self).__init__()

        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode)


class firstOctConv(nn.Module):
    def __init__(self,
                 settings,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(firstOctConv, self).__init__()

        alpha_in, alpha_out = settings
        hf_in_channels = int(in_channels * (1 - alpha_in))
        hf_out_channels = int(out_channels * (1 - alpha_out))
        lf_in_channels = in_channels - hf_in_channels
        lf_out_channels = out_channels - hf_out_channels

        if stride == 2:
            self.pre_pool = nn.Sequential(nn.AvgPool2d(2, 2))
        else:
            self.pre_pool = nn.Sequential()

        self.hf_conv = nn.Conv2d(
            in_channels,
            hf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)

        self.lf_pool = nn.AvgPool2d(2, 2)
        self.lf_conv = nn.Conv2d(
            in_channels,
            lf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        x = self.pre_pool(x)

        out_hf = self.hf_conv(x)
        out_lf = self.lf_conv(self.lf_pool(x))

        return out_hf, out_lf


class lastOctConv(nn.Module):
    def __init__(self,
                 settings,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(lastOctConv, self).__init__()

        alpha_in, alpha_out = settings
        hf_in_channels = int(in_channels * (1 - alpha_in))
        hf_out_channels = int(out_channels * (1 - alpha_out))
        lf_in_channels = in_channels - hf_in_channels
        lf_out_channels = out_channels - hf_out_channels

        if stride == 2:
            self.pre_pool = nn.Sequential(nn.AvgPool2d(2, 2))
        else:
            self.pre_pool = nn.Sequential()

        self.hf_conv = nn.Conv2d(
            hf_in_channels,
            hf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)

        self.lf_conv = nn.Conv2d(
            lf_in_channels,
            hf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        hf_data, lf_data = x

        out_hf = self.hf_conv(self.pre_pool(hf_data))
        out_lf = self.lf_conv(lf_data)
        out = out_hf + out_lf

        return out


class OctConv(nn.Module):
    def __init__(self,
                 settings,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0):
        super(OctConv, self).__init__()

        alpha_in, alpha_out = settings
        hf_in_channels = int(in_channels * (1 - alpha_in))
        hf_out_channels = int(out_channels * (1 - alpha_out))
        lf_in_channels = in_channels - hf_in_channels
        lf_out_channels = out_channels - hf_out_channels

        if stride == 2:
            self.pre_pool = nn.Sequential(nn.AvgPool2d(2, 2))
        else:
            self.pre_pool = nn.Sequential()
        self.hf_conv = nn.Conv2d(
            hf_in_channels,
            hf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)
        self.hf_pool = nn.AvgPool2d(2, 2)
        self.hf_pool_conv = nn.Conv2d(
            hf_in_channels,
            lf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)

        self.lf_conv = nn.Conv2d(
            lf_in_channels,
            hf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)
        if stride == 2:
            self.lf_upsample = nn.Sequential()
            self.lf_down = nn.AvgPool2d(2, 2)
        else:
            # self.lf_upsample = nn.UpsamplingNearest2d(scale_factor=2)
            self.lf_upsample = Sampling(scale_factor=2)
            self.lf_down = nn.Sequential()
        self.lf_down_conv = nn.Conv2d(
            lf_in_channels,
            lf_out_channels,
            kernel_size,
            stride=1,
            padding=padding)

    def forward(self, x):
        hf_data, lf_data = x

        hf_data = self.pre_pool(hf_data)
        hf_conv = self.hf_conv(hf_data)
        hf_pool_conv = self.hf_pool_conv(self.hf_pool(hf_data))

        lf_conv = self.lf_conv(lf_data)
        lf_upsample = self.lf_upsample(lf_conv)
        lf_down_conv = self.lf_down_conv(self.lf_down(lf_data))

        out_hf = hf_conv + lf_upsample
        out_lf = hf_pool_conv + lf_down_conv

        return out_hf, out_lf


class OtcBN(nn.Module):
    def __init__(self, alpha, channels):
        super(OtcBN, self).__init__()

        hf_out_channels = int(channels * (1 - alpha))
        lf_out_channels = channels - hf_out_channels

        self.hf_bn = nn.BatchNorm2d(hf_out_channels)
        self.lf_bn = nn.BatchNorm2d(lf_out_channels)

    def forward(self, x):
        hf_data, lf_data = x

        out_hf = self.hf_bn(hf_data)
        out_lf = self.lf_bn(lf_data)

        return out_hf, out_lf


class OtcAC(nn.Module):
    def __init__(self, alpha, channels):
        super(OtcAC, self).__init__()

        hf_out_channels = int(channels * (1 - alpha))
        lf_out_channels = channels - hf_out_channels

        # using prelu
        self.hf_ac = nn.PReLU(hf_out_channels)
        self.lf_ac = nn.PReLU(lf_out_channels)

    def forward(self, x):
        hf_data, lf_data = x

        out_hf = self.hf_ac(hf_data)
        out_lf = self.lf_ac(lf_data)

        return out_hf, out_lf
########################### OctConv ###########################



########################### Squeeze and Excitation ###########################
class SELayer(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SELayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)
########################### Squeeze and Excitation ###########################



########################### MHE Loss ###########################
class MHELoss(nn.Module):
    def __init__(self, channels, mode='', s='2', eps=1e-4):
        super(MHELoss, self).__init__()

        self.channels = channels
        self.mode = mode
        self.s = s
        self.eps = eps

    def forward(self, kernel):
        kernel = kernel.view(-1, self.channels)

        if self.mode == 'half':
            kernel_neg = -1 * kernel
            kernel = torch.cat((kernel, kernel_neg), dim=-1)
            channels = self.channels * 2
        else:
            channels = self.channels

        cnt = channels * (channels - 1) / 2

        kernel_norm = torch.norm(kernel) + self.eps
        kernel = kernel / kernel_norm
        kernel_mt = torch.mm(torch.transpose(kernel, 0, 1), kernel)
        kernel_mt = torch.clamp(kernel_mt, -1, 1)

        if self.s == '0':
            dis = 2.0 - 2.0 * kernel_mt + 2.0 * torch.diag(torch.diag(kernel_mt))
            loss = -torch.log(dis)
        elif self.s == '1':
            dis = 2.0 - 2.0 * kernel_mt + 2.0 * torch.diag(torch.diag(kernel_mt))
            loss = 1. / torch.sqrt(dis)
        elif self.s == '2':
            dis = 2.0 - 2.0 * kernel_mt + 2.0 * torch.diag(torch.diag(kernel_mt))
            loss = 1. / torch.pow(dis, 1)
        elif self.s == 'a0':
            dis = torch.acos(kernel_mt) / math.pi + self.eps
            loss = -torch.log(dis)
        elif self.s == 'a1':
            dis = torch.acos(kernel_mt) / math.pi + self.eps
            loss = 1. / torch.pow(dis, 1)
        elif self.s == 'a2':
            dis = torch.acos(kernel_mt) / math.pi + self.eps
            loss = 1. / torch.pow(dis, 2)

        loss = torch.triu(loss, diagonal=1)
        loss = 1.0 * torch.sum(loss) / cnt

        return loss


class MHEConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super(MHEConv2d, self).__init__()

        # self.out_channels = out_channels
        if groups == 1:
            self.size = [out_channels, in_channels, kernel_size, kernel_size]
        else:
            self.size = [out_channels, 1, kernel_size, kernel_size]

        self.mheloss = MHELoss(out_channels)

        self.kernel = nn.Parameter(torch.empty(size=self.size))
        nn.init.xavier_uniform_(self.kernel)

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups)
        self.conv.weight = self.kernel

    def forward(self, x):

        x = self.conv(x)
        loss = self.mheloss(self.kernel)

        return x, loss


class MHEDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(MHEDConv2d, self).__init__()

        self.depthwise = MHEConv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = MHEConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        loss = 0.
        x, mhe_loss = self.depthwise(x)
        loss += mhe_loss
        x, mhe_loss = self.pointwise(x)
        loss += mhe_loss

        return x, loss


class MHELinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MHELinear, self).__init__()

        self.out_channels = out_channels
        self.size = [out_channels, in_channels]
        self.kernel = nn.Parameter(torch.empty(size=self.size))
        nn.init.xavier_uniform_(self.kernel)

        self.linear = nn.Linear(in_channels, out_channels)
        self.linear.weight = self.kernel

        self.mheloss = MHELoss(self.out_channels)

    def forward(self, x):
        x = self.linear(x)
        loss = self.mheloss(self.kernel)

        return x, loss
########################### MHE Loss ###########################



########################### Depthwise Coovolution ###########################
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DConv2d, self).__init__()

        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x
########################### Depthwise Coovolution ###########################



########################### Weight Standardization ###########################
class WSConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, scale=5.):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

        self.scale = scale

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                                            keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-8
        weight = weight / std.expand_as(weight)
        weight = weight * self.scale

        return functional.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class WSDConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(WSDConv2d, self).__init__()

        self.depthwise = WSConv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels
                                  )
        self.pointwise = WSConv2d(in_channels, out_channels, 1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


def GroupNorm2d(num_features):
    return nn.GroupNorm(num_channels=num_features, num_groups=num_features // 16)
########################### Weight Standardization ###########################



########################### Adaptive Concat Pool ###########################
class AdaptiveConcatPool2d(nn.Module):
    "Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."
    def __init__(self, sz=None):
        "Output will be 2*sz or 2 if sz is None"
        super().__init__()
        
        self.output_size = sz
        self.ap = nn.AdaptiveAvgPool2d(self.output_size)
        self.mp = nn.AdaptiveMaxPool2d(self.output_size)

    def forward(self, x): 
        
        return torch.cat([self.mp(x), self.ap(x)], 1)
########################### Adaptive Concat Pool ###########################



########################### Normalized Linear ###########################
class NormLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        
        super().__init__(in_features, out_features, bias=False)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, inputs):
        
        return functional.linear(nn.functional.normalize(inputs), nn.functional.normalize(self.weight))
########################### Normalized Linear ###########################