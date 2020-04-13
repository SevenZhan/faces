import torch
from torch import nn
from torch.nn import functional



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