import os
from pathlib import Path
import random
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
import numpy as np
import torch
# 擴增資料
root = Path('/home/ANYCOLOR2434/knife', 'Batch1_NEW')
ls = ['P', 'R', 'B']
dirs = ['train', 'test']
x = []
for i, cls in enumerate(ls):
    data_root = root / cls
    # os.listdir(data_root)
    data = os.listdir(data_root)
    for idx in data:
        img = root / cls / idx / "LReconRaw.Tif"
        x.append(img)
# 建立資料夾
for dir in dirs:
    if not os.path.isdir(dir):
        os.makedirs(dir)
    for path in ls:
        if not os.path.isdir(Path(dir, path)):
            os.makedirs(Path(dir, path))
# set seed
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# 分割 train / test
for link in tqdm(x):
    # set seed
    seed = 2022
    set_seed(seed)
    image = Image.open(link)
    image = Image.fromarray(np.array(image)[27:])
    # image_new = Image.fromarray(np.array(image) - np.min(np.array(image).flatten()))
    image_new = image
    # resize = transforms.Resize([200,200])
    if int(str(link).split('/')[-2]) <= 13:
        b = 0
    elif 13 < int(str(link).split('/')[-2]) <= 18:
        b = 1
    else:
        continue
    angel = 3
    for j in range(100):
        img_aug = transforms.RandomHorizontalFlip()(image_new)
        img_aug = transforms.RandomVerticalFlip()(img_aug)
        img_aug = transforms.RandomRotation(3 + 2 * j)(img_aug)
        # img_aug = transforms.functional.rotate(img_aug, angle=3 + 2 * j)
        img_aug.save(dirs[b] + "/" + str(link).split('/')[-3] + "/" + str(link).split('/')[-2] + "_" + str(j) + ".tif")
    # image_new = resize(image_new)
    # print(image_new.size)

    image_new.save(dirs[b] + "/" + str(link).split('/')[-3] + "/" + str(link).split('/')[-2] + ".tif")