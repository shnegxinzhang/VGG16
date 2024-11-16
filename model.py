import torch.nn as nn
import torch

class VGG16(nn.Module):
    def __init__(self,num_classes=1000,init_weights=True):
        super(VGG16, self).__init__()
        #卷积层，input=224*224*3,output=7*7*512
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3,stride=1, padding=1, bias=False),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,stride=1, padding=1, bias=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3,stride=1, padding=1, bias=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1,bias=False),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        #分类层
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(4096,1000,bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1000,num_classes,bias=False),
        )
        #初始化
        if init_weights:
            self._initialize_weights()
            print('weights initialized')
    #前向传播模型
    def forward(self, x):
        x=self.features(x)
        x=torch.flatten(x,1)
        x=self.classifier(x)
        return x
    #初始化函数，卷积层用kaiming初始化，线性层用正态分布初始化
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias,0)


