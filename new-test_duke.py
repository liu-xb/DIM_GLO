import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import scipy.io as scio
import os
from sklearn import preprocessing
import multiprocessing
import time
import torch.utils.data as data
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(data.Dataset):
    def __init__(self, root, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((root+words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        
    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
 
        if self.transform is not None:
            img = self.transform(img)
        return img,label
    
    def __len__(self):
        return len(self.imgs)
    
    def getName(self):
        return self.classes


tic = time.time()
#######################################################################
######################## Model defination #############################
#######################################################################
class MarketNet(nn.Module):
    def __init__(self):
        super(MarketNet, self).__init__()
        self.resnet_layer = torchvision.models.resnet50(pretrained=False)
        self.pool_bn = nn.BatchNorm1d(self.resnet_layer.fc.in_features)
        self.resnet_layer = nn.Sequential(*list(self.resnet_layer.children())[:-2])
        self.resnet_layer[-1][0].downsample[0]=nn.Conv2d(1024, 2048, kernel_size=(1,1),stride=(1,1), bias=False)
        self.resnet_layer[-1][0].conv2=nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.fc = nn.Linear(2048, 751)

        print('model initilization done')
    
    def forward(self, x):
        x = self.resnet_layer(x)
        self.globalpooling = nn.AvgPool2d(kernel_size=(x.size()[2],x.size()[3]),stride = 1)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        x = self.pool_bn(x)
        return x

transform_test = transforms.Compose([
    transforms.Resize((256,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
queryset = MyDataset(root = '/home/xbliu/disk/duke/',
                    txt = '/home/xbliu/disk/duke/files/query.txt',
                    transform = transform_test)
queryloader = torch.utils.data.DataLoader(queryset, batch_size=10, shuffle=False, num_workers=2)

galleryset = MyDataset(root = '/home/xbliu/disk/duke/',
                    txt = '/home/xbliu/disk/duke/files/gallery.txt',
                    transform = transform_test)
galleryloader = torch.utils.data.DataLoader(galleryset, batch_size=10, shuffle=False, num_workers=2)

################################
def compute_AP(haha):
    good_index = haha[0]
    junk_index = haha[1]
    index = haha[2]
    good_index = good_index[0]
    junk_index = junk_index[0]
    cmc = np.zeros(index.shape[0])
    ngood = good_index.shape[0]
    old_recall = 0
    old_precision = 1.0
    ap = 0
    intersect_size = 0
    j = 0
    good_now = 0
    njunk = 0
    for i in range(index.shape[0]):
        flag = 0
        if index[i] in good_index:
            cmc[i-njunk:] = 1
            flag = 1
            good_now += 1
        if index[i] in junk_index:
            njunk += 1
            continue
        if flag:
            intersect_size += 1
        recall = 1. * intersect_size / ngood
        precision = intersect_size / (j+1.)
        ap += 1.*(recall - old_recall)*((old_precision + precision)/2.)
        old_recall = recall
        old_precision = precision
        j += 1
        if good_now == ngood:
            return cmc[0], cmc[4], cmc[9], ap

def DukeInLineTest(MY_GPU, SNAPSHOT):
    # print 'Using GPU:', str(MY_GPU), ' Snapshot:', SNAPSHOT

    device = torch.device('cuda: '+str(MY_GPU))
    net = MarketNet()
    net.load_state_dict(torch.load(SNAPSHOT), strict=1)
    net = net.to(device)
    with torch.no_grad():
        print('extracting features...')
        querycls = []
        gallerycls = []
        if os.path.exists('querycls.npy'):
            querycls = np.load('querycls.npy')
        else:
            for data in queryloader:
                net.eval()
                images, labels = data
                images = images.to(device)
                outputs1 = net(images)
                outputs1 = outputs1.to('cpu').numpy()#, outputs2.to('cpu').numpy()
                querycls.extend(list(outputs1))
            querycls = preprocessing.normalize( np.array(querycls) )
            np.save('querycls.npy', querycls)

        
        if os.path.exists('gallerycls.npy'):
            gallerycls = np.load('gallerycls.npy')
        else:
            for data in galleryloader:
                net.eval()
                images, labels = data
                images = images.to(device)
                outputs1 = net(images)
                outputs1 = outputs1.to('cpu').numpy()#, outputs2.to('cpu').numpy()
                gallerycls.extend(list(outputs1))
            gallerycls = preprocessing.normalize( np.array(gallerycls) )
            np.save('gallerycls.npy', gallerycls)
    print('feature extraction done')

    mAP = np.zeros(querycls.shape[0])
    CMC = np.zeros((querycls.shape[0], gallerycls.shape[0]))

    index = np.argsort(- np.dot(gallerycls, np.transpose(querycls)), axis = 0)
    good_index = scio.loadmat('duke_evaluation/good_index.mat')['g'][0]
    junk_index = scio.loadmat('duke_evaluation/junk_index.mat')['j'][0]

    pool=multiprocessing.Pool(processes=12)
    result = np.array(pool.map(compute_AP, list(zip(good_index, junk_index, np.transpose(index)))))
    pool.close()
    pool.join()
    result = np.mean(result, axis=0)
    return result[0],result[-1]

acc = DukeInLineTest(0, "camera_gap_0.5-0.761-0.583.pt")
# download camera_gap_0.5-0.761-0.583.pt from https://drive.google.com/file/d/1bDs5YGOg1EdqBpPLhuFAieUvn-NaYWus/view?usp=sharing

print('Rank1:', acc[0], 'mAP:', acc[1])

print(time.time()-tic)
#755-577