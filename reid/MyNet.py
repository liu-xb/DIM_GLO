import torch, torchvision
import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.resnet_layer = torchvision.models.resnet50(pretrained=False)
        self.resnet_layer.load_state_dict(torch.load('/home/xbliu/resnet50-19c8e357.pth'))
        
        self.fc = nn.Linear(2048, 751)
        # self.embedding = nn.Linear(2048, 4096)
        self.pool_bn = nn.BatchNorm1d(2048)
        # self.pool_bn2 = nn.BatchNorm1d(4096)

        nn.init.constant_(self.pool_bn.weight, 1)
        nn.init.constant_(self.pool_bn.bias, 0)
        # nn.init.constant_(self.pool_bn2.weight, 1)
        # nn.init.constant_(self.pool_bn2.bias, 0)
        nn.init.normal_(self.fc.weight, std=0.001)
        nn.init.constant_(self.fc.bias, 0)
        # nn.init.normal_(self.embedding.weight, std=0.001)
        # nn.init.constant_(self.embedding.bias, 0)

        self.resnet_layer = nn.Sequential(*list(self.resnet_layer.children())[:-2])
        self.resnet_layer[-1][0].downsample[0]=nn.Conv2d(1024, 2048, kernel_size=(1,1),stride=(1,1), bias=False)
        self.resnet_layer[-1][0].conv2=nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)

    def forward(self, x):
        x = self.resnet_layer(x)
        self.globalpooling = nn.AvgPool2d(kernel_size=(x.size()[2],x.size()[3]),stride = 1)
        x = self.globalpooling(x)
        x = x.view(x.size(0), -1)
        x2 = self.pool_bn(x)
        # x = self.embedding(x)
        # x2 = self.poolbn2(x)
        y = self.fc(x2)
        return x,x2,y