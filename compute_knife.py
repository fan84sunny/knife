import os
from pathlib import Path
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np


def plot_3d(img):
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # X, Y, Z information
    y, x = img.shape
    X = np.arange(0, x)
    Y = np.arange(0, y)
    X, Y = np.meshgrid(X, Y)
    Z = img
    ax.set_zlim(0, 600)
    # Plot the surface
    surf = ax.plot_surface(
        X, Y, Z,
        cmap='viridis',
        linewidth=0,
        antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=8)
    plt.show()
    # reference: https://matplotlib.org/stable/gallery/mplot3d/surface3d.html


# 計算 Mean , STD
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

# 計算 Mean , STD
image_train = []
image_test = []
image_np = []
image_crop_np = []
for link in tqdm(x):
  image = Image.open(link)
  image_crop400 = np.array(image)[27:][:400]
  image = np.array(image)[27:][:]
  if np.min(np.array(image).flatten()) == 0:
    print('\n有0的連結:', link)

  image_crop_new = (np.array(image_crop400) - np.min(np.array(image_crop400).flatten()))
  image_new = (np.array(image) - np.min(np.array(image).flatten()))
  image_new =image
  # if np.max(image_new)==247:
  #   print('\n有247的連結:',link)
  #   plot_3d(image)
  #   plot_3d(image_new)
  #   np.argwhere(image_new == 247)

  if int(str(link).split('/')[-2]) <= 13:
    b = 0
    image_crop_np.append(image_crop_new)
    image_np.append(image_new)
    # image_train.append(image_new)
  elif 13 < int(str(link).split('/')[-2]) <= 18:
    b = 1
    image_crop_np.append(image_crop_new)
    image_np.append(image_new)
    # image_test.append(image_new)
  else:
    continue

# train_mean = np.mean(image_train.flatten())
# train_std = np.std(image_train.flatten())

# test_mean = np.mean(image_test.flatten())
# test_std = np.std(image_test.flatten())
image_np = np.array(image_np)
image_crop_np = np.array(image_crop_np)

crop_mean = np.mean(image_crop_np)
T_mean = np.mean(image_np.flatten())

crop_std = np.std(image_crop_np)
T_std = np.std(image_np.flatten())

# print('train mean', train_mean,'/t train std', trian_std)
# print('test mean', train_mean,'/t test std', trian_std)
print('total mean', T_mean, '\t total std', T_std)
print('max:', np.max(image_np.flatten()), '\t min:', np.min(image_np.flatten()))

print('total crop mean', crop_mean, '\t total crop std', crop_std)
print('max:', np.max(image_crop_np), '\t min:', np.min(image_crop_np))
