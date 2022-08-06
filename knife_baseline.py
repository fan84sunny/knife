import torch.nn.functional as F
# from random import shuffle
# import pandas as pd
import os
from pathlib import Path
import random
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
import torchvision.models as models
from torch.utils.data import DataLoader
# from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np

# CA (coordinate attention)

import torch
import torch.nn as nn
import math
import torch.nn.functional as F
# from torchsummary import summary
import torch.utils.model_zoo as model_zoo

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
import torch
import torch.nn as nn


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAttention(nn.Module):

    def __init__(self, in_channels, out_channels, reduction=32):
        super(CoordAttention, self).__init__()
        self.pool_w, self.pool_h = nn.AdaptiveAvgPool2d((1, None)), nn.AdaptiveAvgPool2d((None, 1))
        temp_c = max(8, in_channels // reduction)
        self.conv1 = nn.Conv2d(in_channels, temp_c, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(temp_c)
        self.act1 = h_swish()

        self.conv2 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(temp_c, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        short = x
        n, c, H, W = x.shape
        x_h, x_w = self.pool_h(x), self.pool_w(x).permute(0, 1, 3, 2)
        x_cat = torch.cat([x_h, x_w], dim=2)
        out = self.act1(self.bn1(self.conv1(x_cat)))
        x_h, x_w = torch.split(out, [H, W], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        out_h = torch.sigmoid(self.conv2(x_h))
        out_w = torch.sigmoid(self.conv3(x_w))
        return short * out_w * out_h


# 搭建CA_ResNet34
class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None):
        super(BottleneckBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, 1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, 3, padding=dilation, stride=stride, groups=groups, dilation=dilation,
                               bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.ca = CoordAttention(in_channels=planes * self.expansion, out_channels=planes * self.expansion)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.ca(out)  # add CA
        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, depth, n_class=1000, with_pool=True):
        super(ResNet, self).__init__()
        layer_cfg = {
            18: [2, 2, 2, 2],
            34: [3, 4, 6, 3],
            50: [3, 4, 6, 3],
            101: [3, 4, 23, 3],
            152: [3, 8, 36, 3]
        }
        layers = layer_cfg[depth]
        self.num_classes = n_class
        self.with_pool = with_pool
        self._norm_layer = nn.BatchNorm2d

        self.inplanes = 64
        self.dilation = 1

        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        if with_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if n_class > 0:
            self.fc = nn.Linear(512 * block.expansion, n_class)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                norm_layer(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, 1, 64, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.with_pool:
            x = self.avgpool(x)

        if self.num_classes > 0:
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x


def ca_resnet34(**kwargs):
    return ResNet(BottleneckBlock, 34, **kwargs)


def ca_resnet18(**kwargs):
    return ResNet(BottleneckBlock, 18, **kwargs)


def resnet_CA_instance(n_class, pretrained=False, **kwargs):  # resnet18的模型
    model = ResNet(BottleneckBlock, 18, n_class, **kwargs)
    if pretrained:
        pretrained_dict = model_zoo.load_url(model_urls['resnet34'])
        model_dict = model.state_dict()
        # 筛除不加载的层结构
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 更新当前网络的结构字典
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, n_class)  # 15 output classes
    stdv = 1.0 / math.sqrt(1000)
    for p in model.fc.parameters():
        p.data.uniform_(-stdv, stdv)

    return model


# 利用高阶 API 查看模型
# ca_res34 = ca_resnet34(n_class=15)
# print(ca_res34)
# x = torch.rand(1, 3, 224, 224)
# i = ca_res34(x)
# print(i.shape)
# summary(ca_res34, (3, 224, 224))


class TrainDataset(Dataset):
    def __init__(self, train=True, transform=None):
        self.x, self.y = [], []
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
            image = self.transform(Image.fromarray(np.array(image, dtype="float64")))
        return image, self.y[index]


class TestDataset(Dataset):
    def __init__(self, train=False, transform=None):
        self.x, self.y = [], []
        self.transform = transform
        root = Path('/home/ANYCOLOR2434/knife')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        if train:
            b = 0
        else:
            b = 1
        for i, cls in enumerate(ls):
            data_root = root / dirs[b] / cls
            # os.listdir(data_root)
            # data = os.listdir(data_root)
            for idx in range(14, 19):
                file_name = str(idx) + '.tif'
                img = root / dirs[b] / cls / file_name
                self.x.append(img)
                self.y.append(i)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        image = Image.open(self.x[index])
        if self.transform:
            image = self.transform(Image.fromarray(np.array(image, dtype="float64")))
        return image, self.y[index]


# class TestDataset(Dataset):
#     def __init__(self, transform=None):
#         self.x, self.y = [], []
#         self.transform = transform
#         root = Path('/content/drive/MyDrive', 'Batch1')
#         ls = ['P', 'R', 'B']
#         for i, cls in emumerate(ls):
#             data_root = root / cls
#             for idx in os.listdir(data_root)[15:]:
#                 img = root / cls / idx / "LWearDepthRaw.Tif"
#                 self.x.append(img)
#                 self.y.append(i)
#         print(len(self.x))
#     def __len__(self):
#         return len(self.x)
#     def __getitem__(self, index):
#         image = Image.open(self.x[index]).convert('RGB')
#         if self.transform:
#             image = self.transform(image)
#         return image, self.y[index]

def visualization(train, test, title):
    plt.figure()
    plt.plot(train, 'r', label="Train")
    plt.plot(test, 'b', label="Test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(f'curve/{title}-resnet18-xpre-baseline.png')
    plt.show()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False  # 避免卷积不确定
    torch.backends.cudnn.deterministic = True  # 避免不确定算法


def train(epochs=50, lr=1e-4, batch_size=8):
    # set seed
    seed = 2022
    set_seed(seed)
    print('Baseline model training START !!!!!!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODEL

    # ResNet18
    model = models.resnet50(pretrained=False)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features,3)

    # SENet
    # model = torch.hub.load(
    #     'moskomule/senet.pytorch',
    #     'se_resnet20',
    #     num_classes=3)
    # model.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    # ResNeXt 50_32x4d
    # # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)

    # ResNet18+Attn
    # model = ca_resnet18(n_class=3).to(device)

    # model = torch.load('model_save/resnet50.pth')
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[62.5672336])
    ])

    # train
    train_dataset = TrainDataset(train=True, transform=transform)
    test_dataset = TrainDataset(train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, num_workers=8, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=8, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    best = 100
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data).float()

        model.eval()

        test_loss, test_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                test_loss += loss.item()
                test_acc += torch.sum(preds == labels.data).float()

            train_loss = train_loss / len(train_dataset)
            train_acc = train_acc.to('cpu') / len(train_dataset)
            test_loss = test_loss / len(test_dataset)
            test_acc = test_acc.to('cpu') / len(test_dataset)

        scheduler.step(test_loss)

        train_loss_reg.append(train_loss)
        train_acc_reg.append(train_acc)
        test_loss_reg.append(test_loss)
        test_acc_reg.append(test_acc)

        if test_loss < best:
            best_acc = test_acc
            best = test_loss
            torch.save(model, os.path.join('model_save', f'resnet18_xpre_baseline.pth'))

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {test_loss:.4f}\taccuracy: {test_acc:.4f}\n')
    visualization(train_loss_reg, test_loss_reg, 'Loss')
    visualization(train_acc_reg, test_acc_reg, 'Acc')

    model = torch.load(os.path.join('model_save', f'resnet18_xpre_baseline.pth'), map_location=device)
    # visual the test set result
    t_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[62.5672336])
    ])
    t_dataset = TestDataset(train=False, transform=t_transform)
    t_loader = DataLoader(dataset=t_dataset, batch_size=1, shuffle=False)
    dog_probs = []
    model.eval()
    with torch.no_grad():
        ls = ['P', 'R', 'B']
        error_num = 0
        total = 0
        for data, fileid in tqdm(t_loader):
            total += 1
            data = data.to(device)
            fileid = fileid.to(device)
            output = model(data)
            preds = torch.argmax(output, dim=1)
            if preds != fileid.data:
                error_num += 1
                print('\n\n# ', error_num, '/', total)
                print('\n判斷錯誤!')
                print('類別:', ls[fileid.data])
                print('預測:', ls[preds])
                # print('data unsqueeze:', torch.squeeze(data))
                # print('data.data[[]]:',data.data[[]])
                # new_img_PIL = transforms.ToPILImage()(data.data[[]])
                new_img_PIL = transforms.ToPILImage()(torch.squeeze(data))
                plt.imshow(new_img_PIL)
                plt.show()

                # plot_3d(data.numpy())


if __name__ == '__main__':
    train()
    # SteelDataset()
