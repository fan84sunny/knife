import os
import pickle
import time
from pathlib import Path
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import random
from utils import Logger
import sys
from datetime import timedelta, datetime
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import AdaBoostClassifier
# ExtraTreesRegressor,GradientBoostingRegressor, RandomForestRegressor

from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
# Models
# import torchvision.models as models
from models.resnet18_concat_feat import ResNet18, ResBlock

# XGBOOST
import xgboost as xgb


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
        f.write(classification_report(label_list, pred, target_names=targets)+'\n')

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
    torch.backends.cudnn.benchmark = False  # ?????????????????????
    torch.backends.cudnn.deterministic = True  # ?????????????????????
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


def train(save_path, epochs=50, lr=1e-4, batch_size=8, img_size=256):
    print("without initialization")

    # set seed
    seed = 610410113
    set_seed(seed)
    print_cfg(epochs, batch_size, lr, save_path, seed, img_size)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print('[>] Loading dataset '.ljust(64, '-'))
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[1.548061], std=[2.665095]),
        transforms.Normalize(mean=[223.711644], std=[52.5672336]),
    ])
    print("transforms.Normalize(mean=[223.711644], std=[52.5672336])")
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
    ### ResNet18: modified conv1:3*3 without maxpool
    print(model)


    ## model: ResNet18 Attention
    # model = ResNet18_Attn(ResBlock, num_classes=3).to(device)

    ## model: resnet50
    # resnet50 = models.resnet50(pretrained=False)  # ??????????????????????????????????????????
    # resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    # resnet50.maxpool = nn.Identity()
    # model = Net(resnet50)

    print('[*] Model initialized!')
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', verbose=1, patience=3, factor=0.5)
    train_loss_reg, train_acc_reg = [], []
    test_loss_reg, test_acc_reg = [], []
    best = 100
    # best_acc = 0.0
    # best_loss = 0.0
    print('[>] Begin Training '.ljust(64, '-'))

    for epoch in range(epochs):
        train_loss, train_acc = 0.0, 0.0
        print(f'\nEpoch: {epoch + 1}/{epochs}')
        print('-' * len(f'Epoch: {epoch + 1}/{epochs}'))
        model.train()
        for inputs, inputs_flip, labels in tqdm(train_loader):
            labels = labels.to(device)
            inputs_flip = inputs_flip.to(device)
            inputs = inputs.to(device)
            feats, outputs = model(inputs, inputs_flip)
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
                _, outputs = model(inputs, inputs_flip)
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
            _, output = model(data, data_flip)
            preds = torch.argmax(output, dim=1)
            total += 1
            if preds != fileid.data:
                error_num += 1
                print('\n\n# ', error_num, '/', total)
                print('????????????!')
                print('??????:', ls[fileid.data])
                print('??????:', ls[preds])
                # print('data unsqueeze:', torch.squeeze(data))
                # print('data.data[[]]:',data.data[[]])
                # new_img_PIL = transforms.ToPILImage()(data.data[[]])
                # new_img_PIL = transforms.ToPILImage()(torch.squeeze(data))
                # plt.imshow(new_img_PIL)
                # plt.show()


def get_extracted_data(save_path, batch_size=8, img_size=256):
    seed = 610410113
    set_seed(seed)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = torch.load(os.path.join(save_path, f'resnet18_xpre_concat_SP.pth'), map_location=device)
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


    with torch.no_grad():
        for bi, (inputs, inputs_flip, labels) in enumerate(tqdm(train_loader)):
            inputs_flip = inputs_flip.to(device)
            inputs = inputs.to(device)
            feats, outputs = model(inputs, inputs_flip)
            labels = labels.numpy()
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
            if bi == 0:
                test_features = feats.cpu().detach().squeeze(0).numpy()
                test_targets = labels
            else:
                test_features = np.concatenate([test_features, feats.cpu().detach().squeeze(0).numpy()], axis=0)
                test_targets = np.concatenate([test_targets, labels], axis=0)
    print('feat size:',train_features.shape)
    print('feat size:',test_features.shape)
    print('labels size:', train_targets.shape)
    print('labels size:', test_targets.shape)
    return train_features, train_targets, test_features, test_targets
def train_test_ada(save_path, train_feats, train_labels, test_feats, test_labels):
    ada = AdaBoostClassifier(n_estimators=20)
    ada_model_1 = ada.fit(train_feats,train_labels)
    prediction = ada_model_1.predict(test_feats)
    Val_acc = ada_model_1.score(test_feats,test_labels)
    plot_confusion_matrix(save_path, test_labels.reshape(-1), prediction)
    print("Accuarcy: ", Val_acc)
    ### model evaluate
    print("Cohen Kappa quadratic score: ",
          cohen_kappa_score(test_labels, prediction, weights="quadratic"))
    accuracy = accuracy_score(test_labels, prediction)
    print("Accuarcy: %.2f%%" % (accuracy * 100.0))
    plt.savefig(os.path.join(save_path, 'feature_import-resnet18-xpre-AdaBoostClassifier.png'))
    pickle.dump(ada_model_1, open(os.path.join(save_path, "ada_model_1"), "wb"))

def train_test_xgb(save_path, train_feats, train_labels, test_feats, test_labels):
    XGBOOST_PARAM = {
        "random_state": 42,
        'objective': 'multi:softmax',
        "num_class": 3,
        "n_estimators": 200,
        "eval_metric": "mlogloss"
    }
    ### fit model for train data
    xgb_model_1 = xgb.XGBClassifier(**XGBOOST_PARAM)
    xgb_model_1 = xgb_model_1.fit(train_feats, train_labels.reshape(-1),
                                  eval_set=[(test_feats, test_labels.reshape(-1))],
                                  early_stopping_rounds=20,
                                  verbose=False)
    ### make prediction for test data
    prediction = xgb_model_1.predict(test_feats)

    xgb_model_2 = xgb.XGBClassifier(**XGBOOST_PARAM)
    xgb_model_2 = xgb_model_2.fit(test_feats, test_labels.reshape(-1),
                                  eval_set=[(train_feats, train_labels.reshape(-1))],
                                  early_stopping_rounds=20,
                                  verbose=False)
    plot_confusion_matrix(save_path, test_labels.reshape(-1), prediction)
    ### model evaluate
    print("Cohen Kappa quadratic score: ",
          cohen_kappa_score(test_labels, prediction, weights="quadratic"))
    accuracy = accuracy_score(test_labels, prediction)
    print("Accuarcy: %.2f%%" % (accuracy * 100.0))
    ### plot feature importance
    xgb.plot_importance(xgb_model_1, max_num_features=12,title="important features", xlabel='scores', ylabel='features')
    plt.savefig(os.path.join(save_path, 'feature_import-resnet18-xpre-tree.png'))
    # plt.show()

    pickle.dump(xgb_model_1, open(os.path.join(save_path, "xgb_model_1"), "wb"))
    pickle.dump(xgb_model_2, open(os.path.join(save_path, "xgb_model_2"), "wb"))


if __name__ == '__main__':
    current_time = datetime.now().strftime('%b%d_%H-%M-%S') + '_'
    save_path = os.path.join("/home/ANYCOLOR2434/knife/logs", current_time)

    # setting up writers
    sys.stdout = Logger(os.path.join(save_path, 'knife_SP_concat_plusTree.log'))
    img_size = 256
    # -----------------
    start = time.time()
    train(save_path,img_size=img_size)

    # -----------------

    train_feats, train_labels, test_feats, test_labels = get_extracted_data(save_path, img_size=img_size)
    # train_test_xgb(save_path, train_feats, train_labels, test_feats, test_labels)
    train_test_ada(save_path, train_feats, train_labels, test_feats, test_labels)
    # -----------------
    end = time.time()
    # SteelDataset()

    print('\n[*] Finish! '.ljust(64, '-'))
    print(f'[!] total time = {timedelta(seconds=end - start)}s')
    sys.stdout.flush()
