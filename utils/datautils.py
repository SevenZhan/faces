import os
import random
from PIL import Image
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader



# custom dataset api
IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

def has_file_allowed_extension(filename, extensions):

    return filename.lower().endswith(extensions)


def is_image_file(filename):

    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


class Spliter(object):
    def __init__(self, roots, threds=0, valid_pct=0.1):

        self.labels = list()
        self.items = list()

        if not isinstance(roots, list) and not isinstance(roots, tuple):
            if isinstance(threds, list) or isinstance(threds, tuple):
                threds = threds[0]
            self.labels, self.items = self._scandir_(roots, threds)
        elif not isintance(threds, list) and not isinstance(threds, tuple):
            for root in roots:
                results = self._scandir_(root, threds)
                self.labels.extend(results[0])
                self.items.extend(results[1])
        else:
            for root, thred in zip(roots, threds):
                results = self._scandir_(root, thred)
                self.labels.extend(results[0])
                self.items.extend(results[1])

        self.labels2idxs = {self.labels[i]: i for i in range(len(self.labels))}
        random.shuffle(self.items)
        self.validlist, self.trainlist = self.items[:int(len(self.items)*valid_pct)], self.items[int(len(self.items)*valid_pct):]

    @staticmethod
    def _scandir_(root, thred):

        labels = list()
        items = list()

        root = Path(root)
        for base, _, filenames in os.walk(root, followlinks=True):
            if len(filenames) > thred:
                labels.append(base)
                for filename in filenames:
                    filepath = os.path.join(base, filename)
                    if is_image_file(filepath):
                        item = (filepath, base)
                        items.append(item)

        return labels, items

    def __call__(self, transform=None, target_transform=None, batch_size=64, num_workers=0):

        trainset = ImageItemList(self.trainlist, self.labels2idxs, transform=transform[0], target_transform=target_transform)
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)

        validset = ImageItemList(self.validlist, self.labels2idxs, transform=transform[1], target_transform=target_transform)
        validloader = DataLoader(validset, batch_size=batch_size*2, shuffle=True, pin_memory=True, drop_last=True, num_workers=num_workers)

        return trainloader, validloader

class ImageItemList(Dataset):
    def __init__(self, items, labels2idxs, transform=None, target_transform=None):

        self.items = items
        self.labels2idxs = labels2idxs

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):

        return len(self.items)

    def __getitem__(self, index):

        imgpath, target = self.items[index]
        img = Image.open(imgpath).convert('RGB')
        label = self.labels2idxs[target]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label



class RandomBluring(object):

    def __init__(self, p=0.5, high=2):

        self.p = p
        self.high = high

    def __call__(self, img):

        if random.uniform(0, 1) < self.p:
            r = random.randint(1, self.high)
            img = img.filter(ImageFilter.GaussianBlur(radius=r))

        return img
