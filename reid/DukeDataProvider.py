from PIL import Image
import torch.utils.data as data
from .transform_parameter import transform_train,transform_test
import os,random,torch

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset4Auxiliary_GAN(data.Dataset):
    def __init__(self, root1, root2, txt, num_gan_image=1, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((root1+words[0],int(words[1])))
        fh.close()
        GAN_images = os.listdir(root2)
        GAN_images.sort()
        gan_imgs = []
        gan_imgs_index = []
        begin_index = 0
        for i in range(len(imgs)):
            temp_gan_imgs = []
            temp_name = imgs[i][0].split('/')[-1].split('.')[0]        
            for j in range(begin_index,len(GAN_images)):
                if temp_name in GAN_images[j]:
                    temp_gan_imgs.append(root2+GAN_images[j])
                else:
                    begin_index = j
                    break
            gan_imgs.append(temp_gan_imgs)
            gan_imgs_index.append(random.choice(range(len(temp_gan_imgs))))
        self.imgs = imgs
        self.gan_imgs = gan_imgs
        self.gan_imgs_index = gan_imgs_index
        self.loader = loader
        self.num_gan_image = num_gan_image
        
    def __getitem__(self, index):
        fn, _ = self.imgs[index]
        img = self.loader(fn)
        if random.random() > 0.5:
            img = transform_test(img)
        else:
            img = transform_train(img)

        num_gan_image = self.num_gan_image
        if num_gan_image <=0:
            return img, index
        else:
            fn = self.gan_imgs[index]
            fn = fn[self.gan_imgs_index[index]]
            self.gan_imgs_index[index] += 1
            # if self.gan_imgs_index[index] == len(self.gan_imgs[index]):
            #     random.shuffle(self.gan_imgs[index])
            self.gan_imgs_index[index] %= len(self.gan_imgs[index])
            gan_img1 = self.loader(fn)
            gan_img1 = transform_train(gan_img1)
            if num_gan_image == 1:
                return img, gan_img1, index

            fn = self.gan_imgs[index]
            fn = fn[self.gan_imgs_index[index]]
            self.gan_imgs_index[index] += 1
            if self.gan_imgs_index[index] == len(self.gan_imgs[index]):
                random.shuffle(self.gan_imgs[index])
            self.gan_imgs_index[index] %= len(self.gan_imgs[index])
            gan_img2 = self.loader(fn)
            gan_img2 = transform_train(gan_img2)
            if num_gan_image == 2:
                return img, gan_img1, gan_img2, index

            fn = self.gan_imgs[index]
            fn = fn[self.gan_imgs_index[index]]
            self.gan_imgs_index[index] += 1
            if self.gan_imgs_index[index] == len(self.gan_imgs[index]):
                    random.shuffle(self.gan_imgs[index])
            self.gan_imgs_index[index] %= len(self.gan_imgs[index])
            gan_img3 = self.loader(fn)
            gan_img3 = transform_train(gan_img3)
            if num_gan_image == 3:
                return img, gan_img1, gan_img2, gan_img3, index

            fn = self.gan_imgs[index]
            fn = fn[self.gan_imgs_index[index]]
            self.gan_imgs_index[index] += 1
            if self.gan_imgs_index[index] == len(self.gan_imgs[index]):
                    random.shuffle(self.gan_imgs[index])
            self.gan_imgs_index[index] %= len(self.gan_imgs[index])
            gan_img4 = self.loader(fn)
            gan_img4 = transform_train(gan_img4)
            if num_gan_image == 4:
                return img, gan_img1, gan_img2, gan_img3, gan_img4, index
    
    def __len__(self):
        return len(self.imgs)
    
    def getName(self):
        return self.classes

class DukeDataProvider:
    def __init__(self, batch_size, num_gan_image = 1):
      self.batch_size = batch_size
      self.set = MyDataset4Auxiliary_GAN(root1 = '/home/xbliu/disk/duke/',
        root2 = '/home/xbliu/disk/cross_domain_reid/ECN-master/data/duke/bounding_box_train_camstyle/',
        txt = '/home/xbliu/disk/duke/files/train.txt', num_gan_image=num_gan_image)
      self.iter = 0
      self.dataiter = None
    def build(self):
        dataloader = torch.utils.data.DataLoader(self.set,
            batch_size=self.batch_size,shuffle=True, num_workers=8)
        self.dataiter = torch.utils.data.dataloader._MultiProcessingDataLoaderIter(dataloader)
    def next(self):
        if self.dataiter is None:
            self.build()
        try:
            batch = self.dataiter.next()
            self.iter += 1
            return batch
        except StopIteration:
            self.build()
            self.iter=1
            batch = self.dataiter.next()
            return batch