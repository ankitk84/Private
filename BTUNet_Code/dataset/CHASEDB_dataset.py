from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
from os import path
from PIL import Image, ImageFilter

from torch.utils.data import Dataset
from torch.utils.data import Subset
import torch
from torchvision.transforms import functional as TF
from torchvision.datasets.utils import list_files
import sys
sys.path.append('../')
SIZE = 960

class CHASEDB_dataset(Dataset):

    def __init__(self, data_root, train=True, transforms=None):
        super(CHASEDB_dataset, self).__init__()
        self.data_root = data_root
        self.transforms = transforms
        self.num_return = 2
        self.dataset = CHASEDBPILDataset(self.data_root)
        self.train = train
        

    def __getitem__(self, index):
        image, annot = self.dataset[index]

        if self.transforms is None:
            image, annot = self._default_trans(image, annot, self.train)
        else:
            image, annot = self.transforms(image, annot)

        return image, annot

    def __len__(self):
        return len(self.dataset)

    @staticmethod
    def _default_trans(image, annot, train):

        annot = TF.to_grayscale(annot, num_output_channels=1)
        if train:
#             print("trans")
            if random.random() < 0.5:
                image = TF.hflip(image)
                annot = TF.hflip(annot)
            elif random.random() < 0.5:
                image = TF.vflip(image)
                annot = TF.vflip(annot)
            elif random.random() < 0.6:
                angle = random.random() * 360
                image = TF.rotate(img=image, angle=angle)
                annot = TF.rotate(img=annot, angle=angle)

        image = TF.to_tensor(image)
        eps = 1e-8
        mn = torch.min(image)
        mx = torch.max(image)
        image = (image-mn)/(mx-mn+eps)

        annot = TF.to_tensor(annot)
        annot[annot >= 0.5] = 1
        annot[annot < 0.5] = 0
        return image, annot


class CHASEDBPILDataset(Dataset):
    def __init__(self, data_root):
        self.data_root = path.expanduser(data_root)
        self._image_dir = path.join(self.data_root, 'images')
        self._annot_dir = path.join(self.data_root, 'labels')

        self._image_paths = sorted(list_files(self._image_dir, suffix=('.jpg', '.png'), prefix=True))
        self._annot_paths = sorted(list_files(self._annot_dir, suffix=('.png', '.jpg'), prefix=True))

    def __getitem__(self, index):
        image = Image.open(self._image_paths[index], mode='r').convert('RGB')
        image = image.resize((SIZE,SIZE))
        annot = Image.open(self._annot_paths[index], mode='r').convert('1')
        annot = annot.resize((SIZE,SIZE))
        return image, annot

    def __len__(self):
        return len(self._image_paths)


if __name__ == '__main__':
    pass
