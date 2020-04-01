from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

from fastai import *
from fastai.vision import *
from fastai.script import *
from fastai.callbacks import *
from fastai.distributed import *
from fastai.callback.tracker import *

from models.resnets import resnet18
from utils.layerutils import NormLinear
from utils.lossutils import CircleLoss
from utils.datautils import RandomBluring



class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = resnet18()
        self.fc = NormLinear(512, 1000)

    def forward(self, inputs):

        features = self.features(inputs)
        ouputs = self.fc(features)

        return ouputs



@call_parse
def main(gpu=None):
    """Distrubuted training: python -m fastai.launch solver.py"""
    # hparams
    epochs, bs, lr = 15, 256, 1e-1
    mixup = 0.
    # setup gpus
    gpu = setup_distrib(gpu)
    if gpu is None: bs *= torch.cuda.device_count()
    n_gpus = num_distrib() or 1
    workers = min(16, num_cpus()//n_gpus)
    # setup datas
    path = Path('datasets/trainnings/face1000')
    tfms = ([flip_lr(p=0.5)], [])
    data = ImageDataBunch.from_folder(path, valid_pct=0.1, ds_tfms=tfms, bs=bs, num_workers=workers).normalize()
    steps = len(data.train_dl)//n_gpus
    # setup opt, sched, loss
    opt = partial(optim.SGD, momentum=0.9, weight_decay=1e-3)
    phase1 = (TrainingPhase(epochs*0.10*steps).schedule_hp('lr', (lr/10,lr),  anneal=annealing_cos))
    phase2 = (TrainingPhase(epochs*0.90*steps).schedule_hp('lr', (lr,lr/1e5), anneal=annealing_cos))
    loss = CircleLoss()
    # setup learner
    model = Net()
    learn = Learner(data, model, metrics=accuracy, opt_func=opt, loss_func=loss, path='', model_dir='')
    learn.callback_fns += [
        partial(GeneralScheduler, phases=(phase1,phase2)),
        partial(SaveModelCallback, every='epoch', name='model')
    ]
    learn.split([learner.model.features, learner.model.fc])
    if mixup: learn = learn.mixup(alpha=mixup)
    if gpu is None: learn.to_parallel()
    else:           learn.to_distributed(gpu)
    learn.to_fp16(dynamic=True)
    learn.fit(epochs)