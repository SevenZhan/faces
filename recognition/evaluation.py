import os
import argparse
import numpy as np
from PIL import Image

import torch
from torch import nn
from torch.nn import functional
from torchvision import transforms



class Net(nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.arch == 'resnet':
            from fr.models.resnets import Resnet100 as Learner
        else:
            from fr.models.mobiface import Mobiface as Learner

        self.features = Learner()

    def forward(self, inputs):

        return self.features(inputs)



class Extractor(object):
    def __init__(self, opt, checkpoint=None, device=None):

        self.model = Net(opt)
        self.model.to(device)
        self.model.load_state_dict(torch.load(checkpoint, map_location=device)['state_dict'], strict=False)
        self.model.eval()

        self.transformer = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.device = device

    def _features(self, imgpaths):
        imgs = list()
        flips = list()
        for imgpath in imgpaths:
            img = Image.open(imgpath).convert('RGB')
            flip = img.transpose(Image.FLIP_LEFT_RIGHT)

            img = self.transformer(img).numpy()
            flip = self.transformer(flip).numpy()
            imgs.append(img)
            flips.append(flip)

        with torch.no_grad():
            imgs.extend(flips)
            imgs = torch.tensor(imgs).to(self.device)
            features = self.model(imgs).cpu().split(len(imgpaths))
            features = features[0] + features[1]

        return features

    def __call__(self, root, bs=64):

        imgpaths = list()
        labels = list()
        features = list()

        for base, _, imgnames in os.walk(root):
            if len(imgnames):
                label = base.split('/')[-1]
                for imgname in imgnames:
                    imgpath = os.path.join(base, imgname)
                    imgpaths.append(imgpath)
                    labels.append(label)

        for i in range(len(imgpaths)//bs + 1):
            features.extend(self._features(imgpaths[i*bs:(i+1)*bs]))

        return labels, imgpaths, torch.stack(features)



class Evaluator(object):
    def __init__(self, opt, checkpoint, device):

        self.extractor = Extractor(opt, checkpoint, device)

    def _probe(self, root, bs=64):

        self.p_labels, self.p_paths, self.p_features = self.extractor(root, bs)

    def _gallery(self, root, bs=64):

        self.g_labels, self.g_paths, self.g_features = self.extractor(root, bs)

    def _disturb(self, root, bs=64):

        self.d_labels, self.d_paths, self.d_features = self.extractor(root, bs)

    def run(self, k=1000):

        self.g_labels, self.g_features = np.concatenate((self.g_labels, self.d_labels)), torch.cat((self.g_features, self.d_features))

        self.poses = list()
        self.neges = list()

        for p_label, p_feature in zip(self.p_labels, self.p_features):
            p_feature = p_feature.unsqueeze_(0).expand_as(self.g_features)
            similarities = functional.cosine_similarity(p_feature, self.g_features)
            self.poses.append(similarities.masked_select(torch.tensor(self.g_labels == p_label).byte()).numpy())
            neg = similarities.masked_select(torch.tensor(self.g_labels != p_label).byte())
            values, indices = torch.sort(neg, descending=True)
            neg = values[:k]
            self.neges.append(neg.numpy())

        self.poses, self.neges = torch.tensor(self.poses), torch.tensor(self.neges)

    def far(self, far=1e-6):

        n = int(len(self.p_features) * len(self.g_features) * far)
        neges = self.neges.view(-1)
        values, indices = torch.sort(neges, descending=True)
        th = values[n]
        cb = (self.poses > th).float().sum().long()
        print('far@{:.02e} --> [th={:.02f}]: {}, {}'.format(far, th, cb, len(self.poses)))

    def top(self, k=1):

        assert len(self.poses) == len(self.neges)
        cb = list()
        for pos, neg in zip(self.poses, self.neges):
            if k == 1:
                if pos >= neg[0]:
                    cb.append(pos.item())
            else:
                if pos >= neg[:k-1].min():
                    cb.append(pos.item())
        print('top{} --> [th={:.02f}]: {}, {}'.format(k, min(cb), len(cb), len(self.poses)))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', type=str, default='resnet', help='network architecture to use') 
    opt = parser.parse_args()
    
    checkpoint = '/media/disk-3/zhanbin/eval/snapshots/2020-04-25-11-04-39/epoch=06-valid_acc=0.9844.ckpt'
    print('[INFO] checkpoint:', checkpoint.split('/')[-1])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    evaluator = Evaluator(opt, checkpoint, device)

    bs = 512
    disturb = '/media/disk-4/eval/disturb50k'
    print('[INFO] disturb set:', disturb.split('/')[-1])
    evaluator._disturb(disturb, bs)

    probe = '/media/disk-4/eval/mask600/frame'
    gallery = '/media/disk-4/eval/mask600/gallery'
    print('\n[INFO] testing set:', probe.split('/')[-2])
    evaluator._probe(probe, bs)
    evaluator._gallery(gallery, bs)
    evaluator.run(1000)
    evaluator.far(1e-6)
    evaluator.top(1)