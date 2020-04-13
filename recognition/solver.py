import os
import warnings
warnings.simplefilter('ignore')
import datetime
import numpy as np
from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import SGD
from torch.nn import functional
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from utils.lossutils import CircleLoss as LinearFace
from models.resnets import resnet101 as Extractor
from utils.datautils import RandomBluring, RandomErasing

# random seeds
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_data_loaders(train_list, train_bs, valid_list, valid_bs):
    train_ds = ItemList(train_list, transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    train_dl = DataLoader(train_ds,
                          batch_size=train_bs,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True,
                          pin_memory=True
                          )

    valid_ds = ItemList(valid_list, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]))
    valid_dl = DataLoader(valid_ds,
                          batch_size=valid_bs,
                          shuffle=True,
                          num_workers=4,
                          drop_last=True,
                          pin_memory=True
                          )

    return train_dl, valid_dl


class Learner(pl.LightningModule):
    def __init__(self, hparams, checkpoint=None):
        super().__init__()

        self.hparams = hparams

        self.features = Extractor()
        if checkpoint is not None:
            self.features.load_state_dict(checkpoint, strict=False)
        self.fc = LinearFace(512, self.hparams.num_classes)

    def forward(self, x, y=None):
        x = self.features(x)
        p = self.fc(x, y)

        return p

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=self.hparams.lr, momentum=0.9, weight_decay=5e-4)
        scheduler = {'scheduler': MultiStepLR(optimizer, milestones=[5, 8, 11]), 'interval': 'epoch'}

        return [optimizer], [scheduler]

    def prepare_data(self):
        self.trainloader, self.validloader = get_data_loaders(self.hparams.train_list, self.hparams.train_bs,
                                                              self.hparams.valid_list, self.hparams.valid_bs)

    def train_dataloader(self):
        return self.trainloader

    def training_step(self, batch, batch_idx):
        x, y = batch
        p = self(x, y)
        loss = functional.cross_entropy(p, y)

        return {'loss': loss}

    def val_dataloader(self):
        return self.validloader

    def validation_step(self, batch, batch_idx):
        x, y = batch
        p = self(x)
        loss = functional.cross_entropy(p, y)
        acc = p.argmax(dim=-1).eq(y).float().mean()

        return {'loss': loss, 'acc': acc}

    def validation_epoch_end(self, outputs):
        loss = torch.stack([output['loss'] for output in outputs]).mean()
        acc = torch.stack([output['acc'] for output in outputs]).mean()
        tqdm_bar = {'valid_loss': loss, 'valid_acc': acc}

        return {'progress_bar': tqdm_bar}


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_list", type=str, default='ms_train.txt',
                        help="train list containning trainning images")
    parser.add_argument("--train_bs", type=int, default=512, help="input batch size for training (default: 256)")
    parser.add_argument("--valid_list", type=str, default='ms_valid.txt',
                        help="valid list containning validation images")
    parser.add_argument("--valid_bs", type=int, default=512, help="input batch size for validation (default: 512)")
    parser.add_argument("--lr", type=float, default=1e-1, help="max learning rate (default: 0.1)")
    parser.add_argument("--num_classes", type=int, default=78924, help="total number of classes")

    hparams = parser.parse_args()

    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    snapshots = '/mnt/dataserver/zhanbin/snapshots/{}'.format(t)

    learner = Learner(hparams)
    saver = ModelCheckpoint(filepath=os.path.join(snapshots, '{epoch:02d}-{valid_acc:.4f}'), monitor='valid_acc',
                            save_top_k=-1)
    trainer = pl.Trainer(logger=False, checkpoint_callback=saver, gpus=[0, 1], max_epochs=15, distributed_backend='dp',
                         precision=16, benchmark=True)
    trainer.fit(learner)
