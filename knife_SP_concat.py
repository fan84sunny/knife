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
# Models
import torchvision.models as models
from models.resnet18_concat_feat import Net


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


def visualization(train, test, title):
    plt.figure()
    plt.plot(train, 'r', label="Train")
    plt.plot(test, 'b', label="Test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(f'curve/{title}-resnet50-xpre.png')
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


def train(epochs=50, lr=1e-4, batch_size=8):
    # set seed
    seed = 610410113
    set_seed(seed)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[52.5672336])
    ])
    # train
    train_dataset = TrainDataset(train=True, transform=transform)
    val_dataset = TrainDataset(train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=2, batch_size=batch_size, shuffle=True)

    # MODEL
    ## model: ResNet18
    # model = ResNet18(ResBlock, num_classes=3).to(device)

    ## model: ResNet18 Attention
    # model = ResNet18_Attn(ResBlock, num_classes=3).to(device)

    ## model: resnet50
    resnet50 = models.resnet50(pretrained=False)  # 原模型，并加载预训练参数
    resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    resnet50.maxpool = nn.Identity()
    model = Net(resnet50)

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    best = 100
    best_acc = 0.0
    best_loss = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        model.train()
        for inputs, inputs_flip, labels in tqdm(train_loader):
            labels = labels.to(device)
            inputs_flip = inputs_flip.to(device)
            inputs = inputs.to(device)
            outputs = model(inputs, inputs_flip)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            train_loss += loss.item()
            train_acc += torch.sum(preds == labels.data).float()

        model.eval()

        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for inputs, inputs_flip, labels in tqdm(val_loader):
                inputs_flip = inputs_flip.to(device)
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs, inputs_flip)
                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, dim=1)
                val_loss += loss.item()
                val_acc += torch.sum(preds == labels.data).float()

            train_loss = train_loss / len(train_dataset)
            train_acc = train_acc.to('cpu') / len(train_dataset)
            val_loss = val_loss / len(val_dataset)
            val_acc = val_acc.to('cpu') / len(val_dataset)

        scheduler.step(val_loss)

        train_loss_reg.append(train_loss)
        train_acc_reg.append(train_acc)
        test_loss_reg.append(val_loss)
        test_acc_reg.append(val_acc)

        if val_loss < best:
            best = val_loss
            best_acc1 = val_acc
            # torch.save(model, os.path.join('model_save', f'resnet50_xpre_concat_SP.pth'))
        if val_acc < best_acc:
            best_loss = val_loss
            best_acc2 = val_acc
            torch.save(model, os.path.join('model_save', f'resnet50_xpre_concat_SP_TOPacc.pth'))

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {val_loss:.4f}\taccuracy: {val_acc:.4f}\n')
    print(f'Best Val loss: {best:.4f}\taccuracy: {best_acc1:.4f}\n')
    print(f'Best Val loss: {best_loss:.4f}\taccuracy: {best_acc2:.4f}\n')

    visualization(train_loss_reg, test_loss_reg, 'Loss')
    visualization(train_acc_reg, test_acc_reg, 'Acc')
    model = torch.load(os.path.join('model_save', f'resnet50_xpre_concat_SP_TOPacc.pth'), map_location=device)
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
    dog_probs = []
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
