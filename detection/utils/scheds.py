from collections import Counter

from torch.optim.lr_scheduler import _LRScheduler



class WarmupLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1, warmup=5):

        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.warmup = warmup
        super().__init__(optimizer, last_epoch)

    def get_lr(self):

        if self.last_epoch <= self.warmup:
            return [base_lr + self.last_epoch/self.warmup*(base_lr*10 - base_lr) for base_lr in self.base_lrs]
        elif self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma ** self.milestones[self.last_epoch] for group in self.optimizer.param_groups]