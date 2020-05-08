import os
import warnings
warnings.simplefilter('ignore')
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from .utils.lossutils import CircleLoss as LossFunc
from .utils.layerutils import NormLinear
from .models.resnets import resnet101 as Extractor
from .utils.datautils import Spliter, RandomBluring

# random seeds
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)



def evaler(model, valid_dl, loss_fn, device):

    model.eval()
    losses = list()
    acces = list()
    with torch.no_grad():
        for batch in valid_dl:
            inputs, labels = batch[0].to(device), batch[1].to(device)
            ouputs = model(inputs)
            loss = loss_fn(ouputs, labels)
            acc = ouputs.argmax(dim=-1).eq(labels).float().mean()

            losses.append(loss.item())
            acces.append(acc.item())

    return torch.tensor(losses).mean(), torch.tensor(acces).mean()



class Learner(nn.Module):
    def __init__(self, num_classes=1000, checkpoint=None):
        super().__init__()

        self.features = Extractor()
        if checkpoint is not None:
            print('[INFO] Loading from checkpoint...')
            self.features.load_state_dict(checkpoint, strict=False)
        self.fc = NormLinear(512, num_classes)

    def forward(self, inputs):

        features = self.features(inputs)
        ouputs = self.fc(features)

        return ouputs



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--bs', type=int, default=256, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-1, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=15, help='max epochs to train the model')
    parser.add_argument('--print_freq', type=int, default=100, help='steps to print training status')
    hparams = parser.parse_args()

    # config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    snapshots = 'snapshots/{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    if not os.path.exists(snapshots):
        os.makedirs(snapshots)

    # prepare data
    root = ''
    thred = 0
    data = Spliter(root, thred)
    train_tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,)),
        transforms.RandomErasing(p=0.5, scale=(0.1, 0.5), value='random'),
        RandomBluring(p=0.5, high=2)
    ])
    valid_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5,), std=(0.5,))
    ])
    train_dl, valid_dl = data(transform=[train_tfms, valid_tfms], batch_size=hparams.bs, num_workers=4)

    # modeling
    checkpoint = ''
    checkpoint = torch.load(checkpoint, map_location='cpu')
    learner = Learner(len(data.labels), checkpoint)
    learner.to(device)
    learner = nn.DataParallel(learner)

    # optimizing
    loss_fn = LossFunc()
    optimizer = SGD(learner.parameters(), lr=hparams.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 8, 11])

    # training
    for epoch in range(hparams.epochs):
        learner.train()
        for i, batch in enumerate(train_dl):
            step = len(train_dl)*epoch + i + 1
            inputs, labels = batch[0].to(device), batch[1].to(device)
            ouputs = learner(inputs)
            loss = loss_fn(ouputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % hparams.print_freq == 0:
                print('[Iters {:0>7d}/{:0>2d}, lr={:.02e}, {}] loss={:.04f}'.format(step, epoch+1, optimizer.param_groups[0]['lr'], datetime.now().strftime('%H-%M-%S'), loss))
        # evaluating
        print('[INFO] Computer metrics...')
        valid_loss, valid_acc = evaler(learner, valid_dl, loss_fn, device)
        print(' Validation Results - Average Loss: {:.04f} | Accuracy {:.04f}'.format(valid_loss, valid_acc))
        print('[INFO] Complete metrics...')
        scheduler.step()
        # save checkpoint
        print('[INFO] Saving model...')
        torch.save(learner.module.state_dict(), '{}/epoch={}-valid_acc={:.04f}.pth'.format(snapshots, epoch+1, valid_acc))