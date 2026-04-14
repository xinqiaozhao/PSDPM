import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.manifold import TSNE
from data import data_voc, data_coco
from tool import torchutils, pyutils
from tool import torchutils, imutils
from torch.utils.data import DataLoader
import importlib
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import ImageFilter
import random
import numpy as np


img = np.load('exp_coco_without_gt/npy_feat/2011_000513.npy',allow_pickle=True)

img1 = img.item()['feat1']

img2 = torch.Tensor(img1)
img4 = img2[-1,:,:]
img2 = torch.sum(img2,dim=0)
# # print(img2.shape)
# img3 = img2.cpu().numpy()
# fig = plt.figure(figsize=(40,30))
plt.imshow(img2)
plt.axis('off')
plt.show()