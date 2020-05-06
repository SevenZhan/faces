import os
import argparse
import numpy as np

import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import Retina
from valid import evaluate
from utils import TrainDataset, ValidDataset, RandomCroper, RandomFlip, collater, DetectionLosses, WarmupLR



# config
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', type=str, default='wildface', help='root path for training/validation sets')
parser.add_argument('-a', '--arch', type=str, default='resnet50', help='network archtecture to use')
parser.add_argument('-b', '--batch', type=int, default=256, help='batch size to use for trianset/validset')
parser.add_argument('-w', '--worker', type=int, default=4, help='number of workers to use for dataloader')
parser.add_argument('-s', '--states', type=str, default='states', help='path to save model state dicts')
opt = parser.parse_args()



# data preprocessing
train_set = TrainDataset(os.path.join(opt.root, 'train', 'label.txt'), transform=transforms.Compose([RandomCroper(), RandomFlip()]))
train_loader = DataLoader(train_set, batch_size=opt.batch, num_workers=opt.worker, collate_fn=collater, shuffle=True, drop_last=True, pin_memory=True)
valid_set = ValidDataset(os.path.join(opt.root, 'valid', 'label.txt'), transform=transforms.Compose([RandomCroper(), RandomFlip()]))
valid_loader = DataLoader(valid_set, batch_size=opt.batch, num_workers=opt.worker, collate_fn=collater, shuffle=True, drop_last=True, pin_memory=True)



# modeling
model = Retina()
loss_fn = DetectionLosses()
optm_fn = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
scheduler = WarmupLR(optm_fn, [15, 20], warmup=5)



# training
epochs = 25

for epoch in range(epochs):
    for iter, batch in enumerate(train_loader):
        imgs, annotations = batch
        classifications, bboxes, lmarks = model(imgs)
        loss = loss_fn(classifications, bboxes, lmarks, model.anchors, annotations)
        loss.backward()
        optm_fn.zero_grad()
        optm_fn.step()
    scheduler.step()
    recall, precision = evaluate(valid_loader, model)
    torch.save(model.state_dict(), opt.states + '/epoch={}_recall={:.02f}_precision={:.02f}.pth'.format(epoch+1, recall, precision))