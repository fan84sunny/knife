import math
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

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Models
# import torchvision.models as models
# from vit_pytorch import ViT
from models.resnet18_concat_feat import ResNet18, ResBlock
from torch import Tensor


def noisy_tensor(image: Tensor, amount: Tensor, s_vs_p: Tensor, noise_type="SP"):
    if noise_type == "SP":
        mask = torch.ones_like(image) * -1
        # mask = mask.detach().numpy()
        row, col = mask.shape
        row = torch.tensor([row])
        col = torch.tensor([col])
        # Salt mode
        num_salt = torch.ceil(amount * row * col * s_vs_p)
        coords = [torch.randint(0, i - 1, (int(num_salt),))
                  for i in mask.shape]
        mask[tuple(coords)] = 1

        # Pepper mode
        num_pepper = torch.ceil(amount * row * col * (1 - s_vs_p))
        coords = [torch.randint(0, i - 1, (int(num_pepper),))
                  for i in mask.shape]
        mask[tuple(coords)] = 0

        # Onto image
        # mask = torch.from_numpy(mask).detach()
        # print("原圖:\n",image)
        # print("mask:\n",mask)
        image = torch.where(mask == 1, mask, image)
        image = torch.where(mask == 0, mask, image)
        # print("結果:\n")
        return image


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
        out = np.copy(image)
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        gauss = gauss.reshape(row, col)
        noisy = out + gauss
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


def adjust_learning_rate(optimizer, init_lr, epoch, num_epochs, warmup_epochs=10, warmup_lr=1e-6, final_lr=1e-6, warmup=False):
    final_lr = float(final_lr)

    # Setting  schedule function
    if warmup:
        warmup_epochs = warmup_epochs
        warmup_lr = warmup_lr
        if epoch < warmup_epochs:
            cur_lr = warmup_lr + (init_lr - warmup_lr) * ((epoch + 1) / warmup_epochs)
        else:
            cur_lr = final_lr + (init_lr - final_lr) * 0.5 * (
                    1. + math.cos(math.pi * (epoch - warmup_epochs) / (num_epochs - warmup_epochs)))
    else:
        cur_lr = final_lr + (init_lr - final_lr) * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    return cur_lr


def visualization(save_path, train, test, title):
    plt.figure()
    plt.plot(train, 'r', label="Train")
    plt.plot(test, 'b', label="Test")
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.title(title)
    plt.savefig(os.path.join(save_path, f'{title}-resnet18-xpre.png'))
    # plt.show()


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


def print_cfg(epochs, batch, lr, save_path, seed, img_size, warm_up):
    print('[>] Configuration '.ljust(64, '-'))
    print('\tTotal epoch: ', epochs)
    print('\tBatch size: ', batch)
    print('\tlearning rate: ', lr)
    print('\tsave path: ', save_path)
    print('\tseed: ', seed)
    print('\timg_size: ', img_size)
    print('[>] Warmup Setting '.ljust(64, '-'))
    print('\tinit_lr: ', 0.01)
    print('\twarmup_epochs: ', 10)
    print('\twarmup_lr: ', 1e-6)
    print('\tfinal_lr: ', 1e-6)
    print('\twarmup: ', warm_up)

def train(save_path, epochs=50, lr=1e-4, batch_size=8, img_size=224, warm_up=False):
    # set seed
    seed = 610410113
    set_seed(seed)
    print_cfg(epochs, batch_size, lr, save_path, seed, img_size, warm_up)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('[>] Loading dataset '.ljust(64, '-'))
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[52.5672336])
    ])
    print("transforms.Normalize(mean=[223.711644], std=[52.5672336]")
    # train
    train_dataset = TrainDataset(train=True, transform=transform)
    val_dataset = TrainDataset(train=False, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, num_workers=2, batch_size=batch_size, shuffle=True)
    print('[*] Loaded dataset!')

    print('[>] Model '.ljust(64, '-'))
    # MODEL
    ## model: ResNet18
    model = ResNet18(ResBlock, num_classes=3).to(device)

    ## model: ResNet18 Attention
    # model = ResNet18_Attn(ResBlock, num_classes=3).to(device)

    ## model: resnet50
    # resnet50 = models.resnet50(pretrained=False)  # 原模型，並沒有加載預訓練參數
    # resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # resnet50.maxpool = nn.Identity()
    # model = Net(resnet50)

    ## model: ViT
    # patch_size = 32
    # feature_dim = 768
    # layer = 12
    # heads = 12
    # mlp_dim = 3072
    # print("[>] ViT parameter: ",)
    # print("\tpatch_size: ", patch_size,)
    # print("\tfeature_dim: ", feature_dim,)
    # print("\tlayer: ", layer,)
    # print("\theads: ", heads,)
    # print("\theads: ", mlp_dim)

    # ViT_model = ViT(
    #     image_size=img_size,
    #     patch_size=patch_size,
    #     num_classes=3,
    #     dim=feature_dim,
    #     channels=1,
    #     depth=layer,
    #     heads=heads,
    #     mlp_dim=mlp_dim,
    #     dropout=0.1,
    #     emb_dropout=0.1
    # )
    # model = ViT_model

    print('[*] Model initialized!')
    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    best = 100
    # best_acc, best_loss = 0.0, 0.0
    print('[>] Begin Training '.ljust(64, '-'))

    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        model.train()


        for inputs, inputs_flip, labels in tqdm(train_loader):
            adjust_learning_rate(optimizer, 0.01, epoch, epochs, warmup_epochs=10, warmup_lr=1e-6, final_lr=1e-6, warmup=warm_up)
            labels = labels.to(device)
            inputs_flip = inputs_flip.to(device)
            inputs = inputs.to(device)

            # resnet model
            _, outputs = model(inputs, inputs_flip)

            # ViT model
            # outputs = model(inputs)
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
                # resnet model
                _, outputs = model(inputs, inputs_flip)

                # ViT model
                # outputs = model(inputs)

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

            torch.save(model, os.path.join(save_path, f'resnet18_xpre_concat_SP.pth'))
        # if val_acc < best_acc:
        #     best_loss = val_loss
        #     best_acc2 = val_acc
        #     torch.save(model, os.path.join('model_save', f'resnet50_xpre_concat_SP_TOPacc.pth'))

        print(f'Train loss: {train_loss:.4f}\taccuracy: {train_acc:.4f}\n')
        print(f'Test loss: {val_loss:.4f}\taccuracy: {val_acc:.4f}\n')
    print('[>] Best Valid '.ljust(64, '-'))
    stat = (
        f'[+] acc={best_acc1:.4f}\n'
        f'[+] loss={val_loss:.4f}\n'
    )
    print(stat)

    visualization(save_path, train_loss_reg, test_loss_reg, 'Loss')
    visualization(save_path, train_acc_reg, test_acc_reg, 'Acc')
    model = torch.load(os.path.join(save_path, f'resnet18_xpre_concat_SP.pth'), map_location=device)
    # visual the test set result
    t_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[52.5672336]),
        # transforms.Normalize(mean=[1.548061], std=[2.665095]),
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
            _, output = model(data, data_flip)

            # ViT model
            # output = model(data)

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
                # here is show image
                # new_img_PIL = transforms.ToPILImage()(torch.squeeze(data))
                # plt.imshow(new_img_PIL)
                # plt.show()

                # plot_3d(data.numpy())


if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + '_'
    save_path = os.path.join("/home/ANYCOLOR2434/knife/logs", current_time)

    # setting up writers
    sys.stdout = Logger(os.path.join(save_path, 'knife_SP_concat.log'))

    # -----------------
    start = time.time()
    train(save_path, epochs=50, lr=1e-5, batch_size=8, img_size=256, warm_up=False)
    end = time.time()
    # -----------------

    # SteelDataset()

    print('\n[*] Finish! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
