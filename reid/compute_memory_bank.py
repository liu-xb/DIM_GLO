import torch, os, torchvision
from sklearn import preprocessing
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import scipy.io as scio
import torch.nn as nn
import numpy as np
from .MyDataset import MyDataset


def compute_duke_memory_bank(net, device=0, feature_index = 1):
    transform_test = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    DukeTrainSet = MyDataset(root = '/home/xbliu/disk/duke/',
        txt = '/home/xbliu/disk/duke/files/train.txt', transform = transform_test)
    DukeTrainSetLoader = torch.utils.data.DataLoader(DukeTrainSet, batch_size=10, shuffle=False, num_workers=4)

    print('computing duke memory bank')
    with torch.no_grad():
        for i, data in enumerate(DukeTrainSetLoader, 0):
            net.eval()
            if i == 0:
                inputs, label, _ = data
                inputs = inputs.to(device)
                f = net(inputs)[feature_index].detach()
                labels = label
            else:
                inputs, label, _ = data
                inputs = inputs.to(device)
                f = torch.cat((f, net(inputs)[feature_index].detach()), 0)
                labels = torch.cat((labels, label), 0)
    net.train()
    f_norm = f.norm(dim=1, keepdim=True)
    f_norm = f_norm.expand(f.shape[0], f.shape[1])
    f = f / f_norm
    print('done')
    return f, labels

def compute_market_memory_bank(net, device=0, feature_index = 1):
    transform_test = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    DukeTrainSet = MyDataset(root = '/home/xbliu/disk/market1501/',
        txt = '/home/xbliu/disk/market1501/train.txt', transform = transform_test)
    DukeTrainSetLoader = torch.utils.data.DataLoader(DukeTrainSet, batch_size=10, shuffle=False, num_workers=4)

    print('computing market memory bank')
    with torch.no_grad():
        for i, data in enumerate(DukeTrainSetLoader, 0):
            net.eval()
            if i == 0:
                inputs, label,_ = data
                inputs = inputs.to(device)
                f = net(inputs)[feature_index].detach()
                labels = label.to(device)
            else:
                inputs, label,_ = data
                inputs, label = inputs.to(device), label.to(device)
                f = torch.cat((f, net(inputs)[feature_index].detach()), 0)
                labels = torch.cat((labels, label), 0)
    net.train()
    f_norm = f.norm(dim=1, keepdim=True)
    f_norm = f_norm.expand(f.shape[0], f.shape[1])
    f = f / f_norm
    print('done')
    return f, labels