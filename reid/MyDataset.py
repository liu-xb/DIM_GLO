from PIL import Image
import torch.utils.data as data

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
        return img,label,index
    
    def __len__(self):
        return len(self.imgs)
    
    def getName(self):
        return self.classes