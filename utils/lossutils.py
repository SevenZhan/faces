import torch
from torch import nn



class CircleLoss(nn.Module):
    '''
    source: "Circle Loss: A Unified Perspective of Pair Similarity Optimization"
    '''
    def __init__(self, m=0.25, gamma=256):
        
        super().__init__()
        
        self.m = m
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        
        if not self.training:
            return self.loss(inputs*self.gamma, target)

        b = inputs.size(0)
        alpha = torch.clamp_min(inputs + self.m, min=0).detach()
        alpha[range(b), target] = torch.clamp_min(- inputs[range(b), target] + 1 + self.m, min=0).detach()
        delta = torch.ones_like(inputs, device=inputs.device, dtype=inputs.dtype) * self.m
        delta[range(b), target] = 1 - self.m

        return self.loss(alpha * (inputs - delta) * self.gamma, target)



class MrossEntropyLoss(nn.Module):
    def __init__(self, s=32, categ='mos', warmup=True):
        super().__init__()

        self.s = torch.tensor(s)
        self.categ = categ
        self.warmup = warmup
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target):

        if not self.training:
            return self.loss(inputs*self.s, target)

        cos_theta = inputs
        cos_theta = cos_theta.clamp(-1, 1)
        
        b = target.size(0)
        gt = cos_theta[range(b), target].view(-1, 1)  # ground truth score

        if self.categ == 'cos': # cosface
            self.m = torch.tensor(0.35)

            if self.warmup:
                final_gt = torch.where(gt > 0, gt - self.m, gt)
            else:
                final_gt = gt - self.m

        elif self.categ == 'arc':  # arcface
            self.m = torch.tensor(0.50)
            cos_m = torch.cos(self.m)
            sin_m = torch.sin(self.m)
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * cos_m - sin_theta * sin_m  # cos(gt + margin)

            if self.warmup:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m

        elif self.categ == 'mos':
            self.m = torch.tensor(0.35)
            self.t = torch.tensor(0.20)
            mask = cos_theta > gt - self.m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive

            if self.warmup:
                final_gt = torch.where(gt > 0, gt - self.m, gt)
            else:
                final_gt = gt - self.m

        elif self.categ == 'mrc':
            self.m = torch.tensor(0.50)
            self.t = torch.tensor(0.30)
            cos_m = torch.cos(self.m)
            sin_m = torch.sin(self.m)
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * cos_m - sin_theta * sin_m  # cos(gt + margin)
            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = (self.t + 1.0) * hard_vector + self.t  # adaptive

            if self.warmup:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
        else:
            raise Exception('unknown type!')

        cos_theta.scatter_(1, target.view(-1, 1), final_gt)

        return self.loss(cos_theta*self.s, target)