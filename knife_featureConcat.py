import os
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.resnet18_concat_feat import ResNet18, ResBlock

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
            image_flip = self.transform(
                transforms.RandomHorizontalFlip(p=1)(Image.fromarray(np.array(image, dtype="float64"))))
            image = self.transform(Image.fromarray(np.array(image, dtype="float64")))
        return image, image_flip, self.y[index]


class TestDataset(Dataset):
    def __init__(self, train=False, transform=None):
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
            # data_root = root / dirs[b] / cls
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
            image_flip = self.transform(
                transforms.RandomHorizontalFlip(p=1)(Image.fromarray(np.array(image, dtype="float64"))))
            image = self.transform(Image.fromarray(np.array(image, dtype="float64")))
        return image, image_flip, self.y[index]


def visualization(train, test, title):
    plt.figure()
    plt.plot(train, 'r', label="Train")
    plt.plot(test, 'b', label="Test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(f'curve/{title}-resnet18-pre.png')
    plt.show()


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


def train(epochs=50, lr=1e-5, batch_size=4):
    # set seed
    seed = 610410113
    set_seed(seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # MODEL
    ## resnet18
    model = ResNet18(ResBlock, num_classes=3).to(device)

    # resnet18 = models.resnet18(pretrained=False)
    # resnet18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # model = Net(resnet18)
    # model = models.resnet18(pretrained=False)
    # model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # fc_features = model.fc.in_features
    # model.fc = nn.Linear(fc_features, 3)

    ## resnet50
    # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnext50_32x4d', pretrained=True)
    # model = torch.load('model_save/resnet50.pth')

    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[52.5672336])
    ])

    # train
    train_dataset = TrainDataset(train=True, transform=transform)
    test_dataset = TrainDataset(train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, num_workers=2, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    best = 100.0
    best_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        model.train()
        for inputs, inputs_flip, labels in tqdm(train_loader):
            inputs_flip = inputs_flip.to(device)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs, inputs_flip)
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
            for inputs, inputs_flip, labels in tqdm(test_loader):
                inputs_flip = inputs_flip.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs, inputs_flip)
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
            torch.save(model, os.path.join('model_save', f'resnet18_xpre_concat.pth'))

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {test_loss:.4f}\taccuracy: {test_acc:.4f}\n')
    print(f'Best Val loss: {best:.4f}\taccuracy: {best_acc:.4f}\n')


    visualization(train_loss_reg, test_loss_reg, 'Loss')
    visualization(train_acc_reg, test_acc_reg, 'Acc')
    model = torch.load(os.path.join('model_save', f'resnet18_xpre_concat.pth'), map_location=device)
    # visual the test set result
    t_transform = transforms.Compose([
        transforms.Resize([256, 256]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[52.5672336])
    ])
    t_dataset = TestDataset(train=False, transform=t_transform)
    t_loader = DataLoader(dataset=t_dataset, batch_size=1, shuffle=False)
    model.eval()
    with torch.no_grad():
        ls = ['P', 'R', 'B']
        error_num = 0
        total = 0
        for data, data_flip, fileid in tqdm(t_loader):
            data_flip = data_flip.to(device)
            data = data.to(device)
            fileid = fileid.to(device)
            output = model(data, data_flip)
            preds = torch.argmax(output, dim=1)
            total += 1
            if preds != fileid.data:
                error_num += 1
                print('\n\n# ', error_num, '/', total)
                print('判斷錯誤!')
                print('類別:', ls[fileid.data])
                print('預測:', ls[preds])
                new_img_PIL = transforms.ToPILImage()(torch.squeeze(data))
                plt.imshow(new_img_PIL)
                plt.show()

                # plot_3d(data.numpy())


if __name__ == '__main__':
    train()
    # SteelDataset()