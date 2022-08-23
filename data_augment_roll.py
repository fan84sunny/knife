import os
from pathlib import Path
import random
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# set seed
seed = 2022
set_seed(seed)
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
        img = root / cls / idx / "LWearDepthRaw.Tif"
        x.append(img)
# 建立資料夾
for dir in dirs:
    if not os.path.isdir(dir):
        os.makedirs(dir)
    for path in ls:
        if not os.path.isdir(Path(dir, path)):
            os.makedirs(Path(dir, path))
# 分割 train / test
for link in tqdm(x):
    image = Image.open(link)
    image = np.array(image)[27:]
    # image_new = Image.fromarray(np.array(image) - np.min(np.array(image).flatten()))
    image_new = Image.fromarray(image)
    if int(str(link).split('/')[-2]) <= 13:
        b = 0
    elif 13 < int(str(link).split('/')[-2]) <= 18:
        b = 1
    else:
        continue
    angel = 20
    for j in range(1, 51):
        img_aug = np.roll(np.array(image, dtype=float), shift=angel * j, axis=1)
        img_aug = Image.fromarray(img_aug)
        img_aug.save(dirs[b] + "/" + str(link).split('/')[-3] + "/" + str(link).split('/')[-2] + "_" + str(j) + ".tif")
    image_new.save(dirs[b] + "/" + str(link).split('/')[-3] + "/" + str(link).split('/')[-2] + ".tif")
