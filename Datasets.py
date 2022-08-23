import glob
import os
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

def noisy(image, amount=0.004, s_vs_p=0.5, noise_type="SP"):
    # Parameters
    # ----------
    # image : ndarray
    #     Input image data. Will be converted to float.
    # mode : str
    #     One of the following strings, selecting the type of noise to add:

    #     'gauss'     Gaussian-distributed additive noise.
    #     'poisson'   Poisson-distributed noise generated from the data.
    #     's&p'       Replaces random pixels with 0 or 1.
    #     'speckle'   Multiplicative noise using out = image + n*image,where
    #                 n is uniform noise with specified mean & variance.
    if noise_type == "gauss":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = image + gauss
        return noisy
    elif noise_type == "SP":
        row, col = image.shape
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type == "speckle":
        row, col = image.shape
        gauss = np.random.randn(row, col)
        gauss = gauss.reshape(row, col)
        noisy = image + image * gauss
        return noisy


class TrainDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.x, self.y = [], []
        self.transform = transform
        self.train = train
        root = Path('/home/ANYCOLOR2434/knife')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        if train == True:
            b = 0
        else:
            b = 1
        for i, cls in enumerate(ls):
            data_root = root / dirs[b] / cls
            # os.listdir(data_root)
            data = os.listdir(data_root)

            for idx in data:
                img = root / dirs[b] / cls / idx
                self.x.append(img)
                self.y.append(i)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index])
        if self.transform:
            image = Image.fromarray(np.array(image, dtype="float64"))
            if np.random.randint(1, 10) % 2 == 1 & self.train:
                image_np = noisy(np.array(image, dtype="float64"), amount=0.004, s_vs_p=0.5, noise_type="SP")
                image = Image.fromarray(image_np)
            image_flip = self.transform(transforms.RandomHorizontalFlip(p=1)(image))
            image = self.transform(image)
        return image,  self.y[index]

        # return image, image_flip, self.y[index]




class TestDataset(Dataset):
    def __init__(self, train=False, transform=None):
        self.x, self.y = [], []
        self.train = train
        self.transform = transform
        root = Path('/home/ANYCOLOR2434/knife')
        # root = Path('/home/ANYCOLOR2434/knife/Batch1_NEW')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        if train == True:
            b = 0
        else:
            b = 1
        for i, cls in enumerate(ls):
            data_root = root / dirs[b] / cls
            for idx in range(14, 19):
                filename = str(idx) + '.tif'
                img = root / dirs[b] / cls / filename
                self.x.append(img)
                self.y.append(i)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index])
        if self.transform:
            image = Image.fromarray(np.array(image, dtype="float64"))
            image_flip = self.transform(transforms.RandomHorizontalFlip(p=1)(image))
            image = self.transform(image)
        return image, self.y[index]
        # return image, image_flip, self.y[index]


class AllTestDataset(Dataset):
    def __init__(self, train=False, transform=None):
        self.x, self.y = [], []
        self.train = train
        self.transform = transform
        # root = Path('/home/ANYCOLOR2434/knife')
        root = Path('/home/ANYCOLOR2434/knife/Batch1_NEW')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        if train == True:
            b = 0
        else:
            b = 1
        for i, cls in enumerate(ls):
            data_root = root/cls
            data = os.listdir(data_root)
            for idx in data:
                if 1 <= int(idx) <= 15: continue
                img = root / cls / idx / "LWearDepthRaw.Tif"
                self.x.append(img)
                self.y.append(i)

        # for i, cls in enumerate(ls):
        #     data_root = root / dirs[b] / cls
        #     for idx in range(14, 19):
        #         filename = str(idx) + '.tif'
        #         img = root / dirs[b] / cls / filename
        #         self.x.append(img)
        #         self.y.append(i)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index])
        if self.transform:
            image = Image.fromarray(np.array(image, dtype="float64"))
            image_flip = self.transform(transforms.RandomHorizontalFlip(p=1)(image))
            image = self.transform(image)
        return image, self.y[index]
        # return image, image_flip, self.y[index]


