import torch, torchvision
def save_images(a):
    for i in range(len(a)):
        n, c, h, w = a[i].size()
        s = torch.FloatTensor(n,c,h,w).zero_()
        for j in range(n):
            s[j] = a[i][j].data
        s = torchvision.utils.make_grid(s, nrow=5, padding = 30, normalize=True)
        torchvision.utils.save_image(s, str(i)+'.jpg')