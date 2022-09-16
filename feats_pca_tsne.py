import os
import time
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
from utils import Logger
import sys
from datetime import timedelta, datetime
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Models
from models.resnet18_concat_feat import ResNet18, ResBlock
class OriginDataset(Dataset):
    def __init__(self, train=False, transform=None):
        self.x, self.y = [], []
        self.train = train
        self.transform = transform
        root = Path('/home/ANYCOLOR2434/knife')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        if train == True:
            b = 0
        else:
            b = 1
        for i, cls in enumerate(ls):
            data_root = root / dirs[b] / cls
            for idx in range(1, 19):
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
        return image, image_flip, self.y[index]


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
                if "_" in idx:
                    self.y.append(i)
                else:
                    num = ls.index(cls)
                    self.y.append(3+num)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index])
        if self.transform:
            image = Image.fromarray(np.array(image, dtype="float64"))
            # if np.random.randint(1, 10) % 2 == 1 & self.train:
            #     image_np = noisy(np.array(image, dtype="float64"), amount=0.004, s_vs_p=0.5, noise_type="SP")
            #     image = Image.fromarray(image_np)
            image_flip = self.transform(transforms.RandomHorizontalFlip(p=1)(image))
            image = self.transform(image)
        return image, image_flip, self.y[index]


class TestDataset(Dataset):
    def __init__(self, train=False, transform=None):
        self.x, self.y = [], []
        self.train = train
        self.transform = transform
        root = Path('/home/ANYCOLOR2434/knife')
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
        return image, image_flip, self.y[index]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 避免卷积不确定
    torch.backends.cudnn.deterministic = True  # 避免不确定算法
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True)


def print_cfg(epochs, batch, lr, save_path, seed, img_size):
    print('[>] Configuration '.ljust(64, '-'))
    print('\tTotal epoch: ', epochs)
    print('\tBatch size: ', batch)
    print('\tlearning rate: ', lr)
    print('\tsave path: ', save_path)
    print('\tseed: ', seed)
    print('\timg_size: ', img_size)


def get_extracted_data(save_path, model_path, batch_size=8, img_size=256):
    seed = 610410113
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[1.548061], std=[2.665095]),
        transforms.Normalize(mean=[223.711644], std=[52.5672336])
    ])

    train_dataset = TrainDataset(train=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)

    t_dataset = TestDataset(train=False, transform=transform)
    t_loader = DataLoader(dataset=t_dataset, num_workers=2, batch_size=batch_size, shuffle=False)

    model.eval()
    train_features_list = []
    train_label_list = []
    test_features_list = []
    test_label_list = []
    with torch.no_grad():
        for bi, (inputs, inputs_flip, labels) in enumerate(tqdm(train_loader)):
            inputs_flip = inputs_flip.to(device)
            inputs = inputs.to(device)
            feats, outputs = model(inputs, inputs_flip)
            labels = labels.numpy()
            # train_features_list.append(feats[0])
            # train_label_list.append(labels[0])
            if bi == 0:
                train_features = feats.cpu().detach().squeeze(0).numpy()
                train_targets = labels
            else:
                train_features = np.concatenate([train_features, feats.cpu().detach().squeeze(0).numpy()], axis=0)
                train_targets = np.concatenate([train_targets, labels], axis=0)
            # train_feats = np.concatenate([train_feats, feats.cpu().detach().squeeze(0).numpy()], axis=0)
            # print("train_feats shape:",train_feats.shape)
            # train_labels = np.concatenate((train_labels, labels))
            # print("train_labels shape:",train_labels.shape)

        for bi, (data, data_flip, fileid) in enumerate(tqdm(t_loader)):
            data_flip = data_flip.to(device)
            data = data.to(device)
            feats, output = model(data, data_flip)
            labels = fileid.numpy()
            # test_features_list.append(feats[0])
            # test_label_list.append(labels[0])
            if bi == 0:
                test_features = feats.cpu().detach().squeeze(0).numpy()
                test_targets = labels
            else:
                test_features = np.concatenate([test_features, feats.cpu().detach().squeeze(0).numpy()], axis=0)
                test_targets = np.concatenate([test_targets, labels], axis=0)
    print('feat size:', train_features.shape)
    print('feat size:', test_features.shape)
    print('labels size:', train_targets.shape)
    print('labels size:', test_targets.shape)
    # return train_features, train_targets, test_features, test_targets
    return train_features, train_targets, test_features, test_targets


def plot_embedding(save_path, X, y, mode):
    plt.style.use('seaborn-paper')
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['figure.dpi'] = 200
    theme_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ls = ['P', 'R', 'B', 'Orig_P', 'Orig_R', 'Orig_B']
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    fig1 = plt.figure(figsize=(10, 10))

    for i in range(len(ls)):
        feat = X[y == i]
        colors = theme_colors[i]
        plt.scatter(feat[:, 0], feat[:, 1], color=colors, label=ls[i])
    plt.legend()
    fig_name = 'best.png'
    if mode == 'tsne':
        path = os.path.join(save_path, 'tsne')
        if not os.path.isdir(path):
            os.mkdir(path)
    else:
        path = os.path.join(save_path, 'pca')
        if not os.path.isdir(path):
            os.mkdir(path)
    plt.savefig(path + '/' + fig_name)
    print('{} is saved'.format(fig_name))


def compute_tsne_PCA(save_path, features_list, label_list):
    # switch to evaluate mode
    n_iter = 4000
    perplexity = 30
    iterated_power = 300
    print('[>] Configuration '.ljust(64, '-'))
    print('\tTSNE n_iter: ', n_iter)
    print('\tTSNE perplexity: ', perplexity)
    print('\tPCA iterated power: ', iterated_power)

    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', n_iter=n_iter)
    pca = PCA(n_components=2, iterated_power=iterated_power)
    # features_list = torch.stack(features_list)
    # label_list = torch.stack(label_list)
    features_tsne = tsne.fit_transform(features_list)
    plot_embedding(save_path, features_tsne, label_list, 'tsne')
    pca_result = pca.fit_transform(features_list)
    plot_embedding(save_path, pca_result, label_list, 'pca')

if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + '_' + 'feats_visual'
    model_path = '/home/ANYCOLOR2434/knife/logs/Aug27_00-45-48_best/resnet18_xpre_concat_SP.pth'
    save_path = os.path.join("/home/ANYCOLOR2434/knife/logs", current_time)

    # setting up writers
    sys.stdout = Logger(os.path.join(save_path, 'feats_visual_knife_SP_concat.log'))

    # -----------------
    start = time.time()
    train_features, train_targets, test_features, test_targets = get_extracted_data(save_path, model_path, batch_size=8, img_size=256)
    compute_tsne_PCA(save_path, train_features, train_targets)
    # compute_tsne_PCA(save_path, test_features, test_targets)
    end = time.time()
    # -----------------

    # SteelDataset()

    print('\n[*] Finish! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
