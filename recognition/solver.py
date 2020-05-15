import os
import warnings
warnings.simplefilter('ignore')
import numpy as np
from datetime import datetime
from argparse import ArgumentParser

import torch
from torch import nn
from torch.optim import SGD
from torchvision import transforms
from torch import distributed as dist
from torch import multiprocessing as mp
from torch.optim.lr_scheduler import MultiStepLR
from torch.nn.parallel import DistributedDataParallel as DDP

from .utils.layerutils import NormLinear
from .models.resnets import resnet101 as Extractor
from .utils.lossutils import CircleLoss as LossFunc
from .utils.datautils import Spliter, RandomBluring



# configs
SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True



class Learner(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.features = Extractor()
        self.fc = NormLinear(512, num_classes)

    def forward(self, inputs):

        features = self.features(inputs)
        ouputs = self.fc(features)

        return ouputs



def train(model, train_dl, criterion, optimizer, epoch, args):

    model.train()
    for i, batch in enumerate(train_dl):
        step = len(train_dl) * epoch + i + 1
        inputs, labels = batch[0].cuda(args.gpu, non_blocking=True), batch[1].cuda(args.gpu, non_blocking=True)
        ouputs = model(inputs, labels)
        loss = criterion(ouputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.rank == 0 and step % args.print_freq == 0:
            print('[Iters {:0>7d}/{:0>2d}, lr={:.02e}, {}] loss={:.04f}'.format(step, epoch + 1, optimizer.param_groups[0]['lr'], datetime.now().strftime('%H-%M-%S'), loss))



def validate(model, valid_dl, criterion, args):

    model.eval()
    loss = 0.
    acc = 0.
    with torch.no_grad():
        for batch in valid_dl:
            inputs, labels = batch[0].cuda(args.gpu, non_blocking=True), batch[1].cuda(args.gpu, non_blocking=True)
            ouputs = model(inputs)
            loss += criterion(ouputs, labels)
            acc += ouputs.argmax(dim=-1).eq(labels).float().mean()

    return loss/len(valid_dl), acc/len(valid_dl)



def engine(gpu, args):

    args.gpu = gpu
    args.rank = args.nr*args.gpus + gpu
    args.workers = int((args.workers + args.gpus - 1) / args.gpus)
    dist.init_process_group(backend='nccl', init_method='env://', world_size=args.world_size, rank=args.rank)
    # prepare data
    data = Spliter(args.root, args.thred)
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
    train_dl, valid_dl, train_sp = data(transform=[train_tfms, valid_tfms], batch_size=args.batch_size, num_workers=args.workers)
    # modeling
    model = Learner(num_classes=len(data.labels))
    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    # transfer learning
    if not args.checkpoint is None:
        print('[INFO] Loading from checkpoint...')
        model.load_state_dict(torch.load(args.checkpoint, map_location='cpu'))
    model = DDP(model, device_ids=[gpu])
    # define loss function (criterion) and optimizer
    criterion = LossFunc()
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[5, 8, 11])
    # training
    snapshots = 'snapshots/{}'.format(datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
    for epoch in range(args.epochs):
        train_sp.set_epoch(epoch)
        train(model, train_dl, criterion, optimizer, epoch, args)
        scheduler.step()
        if args.rank == 0:
            # validation
            print('[INFO] Computer metrics...')
            valid_loss, valid_acc = validate(model, valid_dl, criterion, args)
            print(' Validation Results - Average Loss: {:.04f} | Accuracy {:.04f}'.format(valid_loss, valid_acc))
            print('[INFO] Complete metrics...')
            # checkpointing
            print('[INFO] Saving model...')
            if not os.path.exists(snapshots):
                os.makedirs(snapshots)
            torch.save(model.module.state_dict(), '{}/epoch={:0>2d}-valid_acc={:.04f}.pth'.format(snapshots, epoch+1, valid_acc))
    # finalize
    dist.destroy_process_group()



if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int)
    parser.add_argument('-nr', '--nr', default=0, type=int, help='ranking within the nodes')

    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
    parser.add_argument('--lr', default=1e-1, type=float, help='initial learning rate')
    parser.add_argument('--epochs', default=15, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')

    parser.add_argument('--root', type=str, default='./data', help='path to data')
    parser.add_argument('--thred', type=int, default=0, help='minimal number of samples per class')

    args = parser.parse_args()
    args.gpus = torch.cuda.device_count()
    args.world_size = args.gpus * args.nodes
    args.checkpoint = None

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '8888'
    mp.spawn(engine, nprocs=args.gpus, args=(args,))