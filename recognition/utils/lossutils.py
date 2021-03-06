import torch
from torch import nn



########################### Circle Loss ###########################
class CircleLoss(nn.Module):
    '''
    source: "Circle Loss: A Unified Perspective of Pair Similarity Optimization."
    '''
    def __init__(self, m=0.25, gamma=256):
        
        super().__init__()
        
        self.m = torch.tensor(m)
        self.gamma = torch.tensor(gamma)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, target):
        
        if not self.training:
            return self.loss(inputs*self.gamma, target)

        b = inputs.size(0)
        alpha = torch.clamp_min(inputs + self.m, min=0)
        alpha[range(b), target] = torch.clamp_min(-inputs[range(b), target] + 1 + self.m, min=0).type_as(inputs)
        delta = torch.ones_like(inputs, device=inputs.device, dtype=inputs.dtype) * self.m
        delta[range(b), target] = (1 - self.m).type_as(inputs)

        return self.loss(alpha * (inputs - delta) * self.gamma, target)
########################### Circle Loss ###########################



########################### Mis-classified Vector Guided Softmax ###########################
class MrossEntropyLoss(nn.Module):
    '''
    source: "Mis-classified Vector Guided Softmax Loss for Face Recognition."
    '''
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
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2)).type_as(cos_theta)
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
            cos_theta[mask] = ((self.t + 1.0) * hard_vector + self.t).type_as(cos_theta)  # adaptive

            if self.warmup:
                final_gt = torch.where(gt > 0, gt - self.m, gt)
            else:
                final_gt = gt - self.m

        elif self.categ == 'mrc':
            self.m = torch.tensor(0.50)
            self.t = torch.tensor(0.30)
            cos_m = torch.cos(self.m)
            sin_m = torch.sin(self.m)
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2)).type_as(cos_theta)
            cos_theta_m = gt * cos_m - sin_theta * sin_m  # cos(gt + margin)
            mask = cos_theta > cos_theta_m
            hard_vector = cos_theta[mask]
            cos_theta[mask] = ((self.t + 1.0) * hard_vector + self.t).type_as(cos_theta)  # adaptive

            if self.warmup:
                final_gt = torch.where(gt > 0, cos_theta_m, gt)
            else:
                final_gt = cos_theta_m
        else:
            raise Exception('unknown type!')

        cos_theta.scatter_(1, target.view(-1, 1), final_gt)

        return self.loss(cos_theta*self.s, target)
########################### Mis-classified Vector Guided Softmax ###########################



########################### Curricular Face ###########################
class CurricularFace(nn.Module):
    '''
    source: "CurricularFace: Adaptive Curriculum Learning Loss for Deep Face Recognition."
    '''
    def __init__(self, s=64., m=0.5, t=1.0, alpha=0.01):
        super().__init__()

        self.s = torch.tensor(s)
        self.m = torch.tensor(m)
        self.t = torch.tensor(t)
        self.alpha = torch.tensor(alpha)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):

        if not self.training:
            return self.loss(self.s*inputs, labels)

        cos_theta = inputs
        cos_theta = cos_theta.clamp(-1, 1)
        cos_m, sin_m = torch.cos(self.m), torch.sin(self.m)

        target_logit = cos_theta[torch.arange(0, inputs.size(0)), labels].view(-1, 1)
        sin_theta = torch.sqrt(1.0 - torch.pow(target_logit, 2))
        cos_theta_m = target_logit * cos_m - sin_theta * sin_m

        with torch.no_grad():
            self.t = target_logit.mean() * self.alpha + (1 - self.alpha) * self.t

        hard_mask = cos_theta > cos_theta_m
        cos_theta[hard_mask] *= self.t + cos_theta[hard_mask]
        cos_theta.scatter_(1, labels.view(-1, 1), cos_theta_m)

        return self.loss(self.s*cos_theta, labels)
########################### Curricular Face ###########################



########################### Label Smoothing ###########################
class LabelSmoothing(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()

        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):

        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss

        return loss.mean()
########################### Label Smoothing ###########################



########################### Unknown Identity Rejection ###########################
class UIRLoss(nn.Module):
    '''
    source: "Unknown Identity Rejection Loss: Utilizing Unlabeled Data for Face Recognition."
    '''
    def __init__(self):
        super().__init__()

        self.softmax = nn.Sequential(
            nn.Softmax(dim=-1),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, inputs):

        outs = self.softmax(inputs)
        outs = torch.sum(outs, dim=-1)
        assert outs.size(0) == inputs.size(0)

        return -outs.mean()
########################### Unknown Identity Rejection ###########################