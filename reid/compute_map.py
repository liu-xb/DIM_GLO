import numpy as np
import torchvision.transforms as transforms
from sklearn import preprocessing
import multiprocessing
from .MyDataset import MyDataset
import torch
import scipy.io as scio

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
            return cmc[0], cmc[5], cmc[10], ap

def SearchMarket(net, device, feature_index):
    transform_search = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    queryset = MyDataset(root = '/home/xbliu/disk/market1501/',
                        txt = '/home/xbliu/disk/market1501/query.txt',
                        transform = transform_search)
    queryloader = torch.utils.data.DataLoader(queryset, batch_size=10, shuffle=False, num_workers=2)

    galleryset = MyDataset(root = '/home/xbliu/disk/market1501/',
                        txt = '/home/xbliu/disk/market1501/gallery.txt',
                        transform = transform_search)
    galleryloader = torch.utils.data.DataLoader(galleryset, batch_size=10, shuffle=False, num_workers=2)
    with torch.no_grad():
        queryfeature = []
        for data in queryloader:
            net.eval()
            images, _, _ = data
            images = images.to(device)
            outputs = net(images)
            feature = outputs[feature_index[0]]
            for i in range(len(feature_index)-1):
                feature = torch.cat((feature,outputs[feature_index[i+1]]), 1)
            feature  = feature.to('cpu').numpy()
            queryfeature.extend(list(feature))
        queryfeature = preprocessing.normalize( np.array(queryfeature) )

        galleryfeature = []
        for data in galleryloader:
            net.eval()
            images, labels,_ = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            feature = outputs[feature_index[0]]
            for i in range(len(feature_index)-1):
                feature = torch.cat((feature,outputs[feature_index[i+1]]), 1)
            feature  = feature.to('cpu').numpy()
            galleryfeature.extend(list(feature))
        galleryfeature = preprocessing.normalize( np.array(galleryfeature) )
    index = np.argsort(- np.dot(galleryfeature, np.transpose(queryfeature)), axis = 0) + 1
    good_index = scio.loadmat('evaluation/good_index.mat')['good_index'][0]
    junk_index = scio.loadmat('evaluation/junk_index.mat')['junk_index'][0]
    
    pool=multiprocessing.Pool(processes=6)
    result = np.array(pool.map(compute_AP, list(zip(good_index, junk_index, np.transpose(index)))))
    pool.close()
    pool.join()
    result = np.mean(result, axis=0)
    # print 'R1:', "%.3f" % result[0], ' mAP:', '%.3f' % result[-1], '\n'
    return result[0],result[-1]

def SearchDuke(net, device, feature_index):
    transform_search = transforms.Compose([
        transforms.Resize((256,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    queryset = MyDataset(root = '/home/xbliu/disk/duke/',
                        txt = '/home/xbliu/disk/duke/files/query.txt',
                        transform = transform_search)
    queryloader = torch.utils.data.DataLoader(queryset, batch_size=10, shuffle=False, num_workers=2)

    galleryset = MyDataset(root = '/home/xbliu/disk/duke/',
                        txt = '/home/xbliu/disk/duke/files/gallery.txt',
                        transform = transform_search)
    galleryloader = torch.utils.data.DataLoader(galleryset, batch_size=10, shuffle=False, num_workers=2)
    with torch.no_grad():
        querycls = []
        gallerycls = []
        for data in queryloader:
            net.eval()
            images, _,_ = data
            images = images.to(device)
            outputs = net(images)
            feature = outputs[feature_index[0]]
            for i in range(len(feature_index)-1):
                feature = torch.cat((feature,outputs[feature_index[i+1]]), 1)
            feature  = feature.to('cpu').numpy()
            querycls.extend(list(feature))
        querycls = preprocessing.normalize( np.array(querycls) )
        for data in galleryloader:
            net.eval()
            images, _,_ = data
            images = images.to(device)
            outputs = net(images)
            feature = outputs[feature_index[0]]
            for i in range(len(feature_index)-1):
                feature = torch.cat((feature,outputs[feature_index[i+1]]), 1)
            feature  = feature.to('cpu').numpy()
            gallerycls.extend(list(feature))
        gallerycls = preprocessing.normalize( np.array(gallerycls) )

    index = np.argsort(- np.dot(gallerycls, np.transpose(querycls)), axis = 0)
    good_index = scio.loadmat('duke_evaluation/good_index.mat')['g'][0]
    junk_index = scio.loadmat('duke_evaluation/junk_index.mat')['j'][0]

    pool=multiprocessing.Pool(processes=12)
    result = np.array(pool.map(compute_AP, list(zip(good_index, junk_index, np.transpose(index)))))
    pool.close()
    pool.join()
    result = np.mean(result, axis=0)
    return result
