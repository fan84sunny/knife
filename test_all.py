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
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def plot_confusion_matrix(save_path, label_list, y_pred_list):
    pred = y_pred_list
    confusion_mat = confusion_matrix(label_list, pred)
    targets = ['P', 'R', 'B']
    # Visualize confusion matrix
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(confusion_mat, interpolation='nearest', cmap=plt.cm.Blues, alpha=0.3)
    for i in range(confusion_mat.shape[0]):
        for j in range(confusion_mat.shape[1]):
            ax.text(x=j, y=i, s=confusion_mat[i, j], va='center', ha='center')
    plt.title('Confusion matrix')
    # plt.colorbar()
    # ticks = np.arange(3)
    # plt.xticks(ticks, ticks)
    # plt.yticks(ticks, ticks)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    # plt.show()

    fig_name = 'confusion matrix.png'

    plt.savefig(save_path + '/' + fig_name)
    print('{} is saved'.format(fig_name))

    # Classification report
    # print('\n', classification_report(label_list, pred, target_names=targets))
    cls_report_name = 'classification_report.txt'
    with open(save_path + '/' + cls_report_name, "a+") as f:
        f.write(classification_report(label_list, pred, target_names=targets) + '\n')
class TrainDataset(Dataset):
    def __init__(self, train=True, transform=None):
        print('All Val Dataset:P 5*51 張、R 5*51 張、B 5*51 張')

        self.x, self.y = [], []
        self.transform = transform
        self.train = train
        root = Path('/home/ANYCOLOR2434/knife')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        # dirs = ['_train', '_val', '_test']

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
            if np.random.randint(1, 10) % 2 == 1 and self.train:
                image_np = noisy(np.array(image, dtype="float64"), amount=0.004, s_vs_p=0.5, noise_type="SP")
                image = Image.fromarray(image_np)
            image_flip = self.transform(transforms.RandomHorizontalFlip(p=1)(image))
            image = self.transform(image)
        return image, image_flip, self.y[index]

class TestDataset(Dataset):
    def __init__(self, train=False, transform=None):
        print('TestDataset: 5pcs/per cls')
        self.x, self.y = [], []
        self.train = train
        self.transform = transform
        root = Path('/home/ANYCOLOR2434/knife')
        ls = ['P', 'R', 'B']
        dirs = ['train', 'test']
        # dirs = ['_train', '_val', '_test']

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


class ALLTestDataset(Dataset):
    def __init__(self, train=False, transform=None):
        print('All test Dataset:P 5 張、R 17 張、B 33 張')

        self.x, self.y = [], []
        self.train = train
        self.transform = transform

        root = Path('/home/ANYCOLOR2434/knife', 'Batch1_NEW')
        ls = ['P', 'R', 'B']
        # dirs = ['_train', '_val', '_test']
        # for i, cls in enumerate(ls):
        #     data_root = root / cls
        #     # os.listdir(data_root)
        #     data = os.listdir(data_root)
        #     for idx in data:
        #         img = root / cls / idx / "LWearDepthRaw.Tif"
        #         x.append(img)
        for i, cls in enumerate(ls):
            data_root = root / cls
            data = os.listdir(data_root)
            for idx in data:
                if 13 < int(idx):
                    img = root / cls / idx / "LWearDepthRaw.Tif"
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


def ALLtest(save_path, model_path, epochs=50, lr=1e-4, batch_size=8, img_size=224, warm_up=False):
    # print('This is for aug validation!')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t_transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[223.711644], std=[52.5672336]),
        # transforms.Normalize(mean=[1.548061], std=[2.665095]),
    ])
    # val_dataset = TrainDataset(train=False, transform=t_transform)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    test_dataset= TestDataset(train=False, transform=t_transform)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    # t_dataset = ALLTestDataset(train=False, transform=t_transform)
    # t_loader = DataLoader(dataset=t_dataset, batch_size=1, shuffle=False)
    print('[>] Model '.ljust(64, '-'))

    # MODEL
    model = torch.load(model_path, map_location=device)
    # visual the test set result
    model.eval()
    print('[*] Model initialized!')

    print('[>] Begin Testing '.ljust(64, '-'))

    y_pred = []
    with torch.no_grad():
        ls = ['P', 'R', 'B']
        error_num = 0
        total = 0
        for bi, (data, data_flip, fileid) in enumerate(tqdm(test_loader)):
            data_flip = data_flip.to(device)
            data = data.to(device)
            fileid = fileid.to(device)
            _, output = model(data, data_flip)
            # fileid = fileid.numpy()
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

            y_pred.append(preds)
            fileid = fileid.cpu().numpy()
            if bi == 0:
                test_targets = fileid
            else:
                test_targets = np.concatenate([test_targets, fileid], axis=0)

        plot_confusion_matrix(save_path, test_targets.reshape(-1), torch.cat(y_pred, 0).cpu().numpy())


if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + '_'
    save_path = os.path.join("/home/ANYCOLOR2434/knife/logs", current_time)

    # setting up writers
    sys.stdout = Logger(os.path.join(save_path, 'knife_SP_concat_noSP5test.log'))
    model_path = '/home/ANYCOLOR2434/knife/logs/Sep17_22-28-32_BESTnoSP/resnet18_xpre_concat_SP.pth'
    print(model_path)
    # -----------------
    start = time.time()
    ALLtest(save_path, model_path=model_path,
            epochs=50, lr=1e-4, batch_size=8, img_size=256, warm_up=False)
    end = time.time()
    # -----------------

    # SteelDataset()

    print('\n[*] Finish! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
