import os
import warnings
warnings.simplefilter('ignore')
import datetime
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from .utils.lossutils import CircleLoss as LossFunc
from .utils.layerutils import NormLinear
from .models.resnets import resnet101 as Extractor
from .utils.datautils import Spliter, RandomBluring

# random seeds
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)



class Learner(pl.LightningModule):
    def __init__(self, hparams, checkpoint=None):
        super().__init__()

        self.hparams = hparams
        self.data = Spliter(self.hparams.roots, self.hparams.threds)

        self.features = Extractor()
        self.fc = NormLinear(512, len(self.data.labels))
        if checkpoint is not None:
            self.load_state_dict(checkpoint)

        self.loss = LossFunc()

    def forward(self, x):

        x = self.features(x)
        p = self.fc(x)

        return p

    def configure_optimizers(self):

        optimizer = SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = {'scheduler': MultiStepLR(optimizer, milestones=[5, 8, 11]), 'interval': 'epoch'}

        return [optimizer], [scheduler]

    def prepare_data(self):

        bs = self.hparams.bs

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
        self.trainloader, self.validloader = self.data(transform=[train_tfms, valid_tfms], batch_size=bs, num_workers=4)

    def train_dataloader(self):

        return self.trainloader

    def training_step(self, batch, batch_idx):

        x, y = batch
        p = self(x)
        loss = self.loss(p, y)

        return {'loss': loss}

    def val_dataloader(self):

        return self.validloader

    def validation_step(self, batch, batch_idx):

        x, y = batch
        p = self(x)
        loss = self.loss(p, y)
        acc = p.argmax(dim=-1).eq(y).float().mean()

        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):

        loss = torch.stack([output['loss'] for output in outputs]).mean()
        acc = torch.stack([output['acc'] for output in outputs]).mean()
        tqdm_bar = {'valid_loss': loss, 'valid_acc': acc}

        return {'progress_bar': tqdm_bar}


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--roots", type=str, default='', help="path to training/validation images")
    parser.add_argument("--threds", type=int, default=0, help="threshold for valid training/validation images folder")
    parser.add_argument("--bs", type=int, default=512, help="input batch size for training")
    parser.add_argument("--lr", type=float, default=1e-1, help="max learning rate")

    hparams = parser.parse_args()

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    snapshots = 'snapshots/{}'.format(t)

    learner = Learner(hparams)
    saver = ModelCheckpoint(filepath=os.path.join(snapshots, '{epoch:02d}-{valid_acc:.4f}'), monitor='valid_acc',
                            save_top_k=-1)
    trainer = pl.Trainer(logger=False, checkpoint_callback=saver, gpus=[0, 1], max_epochs=15, distributed_backend='dp',
                         precision=32, benchmark=True)
    trainer.fit(learner)
