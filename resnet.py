import torch
import torch.nn as nn
import torch.nn.functional as F


# 定義殘差塊ResBlock
class ResBlock(nn.Module):
    """source: https://www.twblogs.net/a/5bfacbddbd9eee7aed32beda"""
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResBlock, self).__init__()
        # 這裡定義了殘差塊內連續的2個卷積層
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            # shortcut，這里為了跟2個卷積層的結果結構一致，要做處理
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        # 將2個卷積層的輸出跟處理過的x相加，實現ResNet的基本結構
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """source: https://www.twblogs.net/a/5bfacbddbd9eee7aed32beda"""
    def __init__(self, ResBlock, num_classes=3):
        super(ResNet18, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer1 = self.make_layer(ResBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResBlock, 512, 2, stride=2)
        self.fc = nn.Linear(102400, num_classes)

    # 這個函數主要是用來，重複同一個殘差塊
    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x1, x2):
        # input 2 image convert to feature
        out1 = self.conv1(x1)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.layer4(out1)
        out1 = F.avg_pool2d(out1, 3)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.conv1(x2)
        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2 = F.avg_pool2d(out2, 3)
        out2 = out2.view(out2.size(0), -1)
        # concat 2 features
        out = torch.concat((out1, out2), 1)

        out = self.fc(out)
        return out


#####Resnet50
import torchvision.models as models


# model = models.resnet50(pretrained=True)
# fc_features = model.fc.in_features
# model.fc = nn.Linear(fc_features, 3)
# resnet_layer = nn.Sequential(*list(model.children())[:-2])
class Net(nn.Module):
    def __init__(self, model):  # 此处的model参数是已经加载了预训练参数的模型，方便继承预训练成果
        super(Net, self).__init__()
        # 取掉model的后两层
        # self.resnet_layer = nn.Sequential(*list(model.children())[:-1])
        self.resnet_layer = nn.Sequential(*list(model.children())[:-2])

        # self.transion_layer = nn.ConvTranspose2d(2048, 2048, kernel_size=14, stride=3)
        # self.pool_layer = nn.MaxPool2d(32)
        self.fc = nn.Linear(3211264, 3)

    def forward(self, x1, x2):
        # 在这里，整个ResNet18的结构就很清晰了
        out1 = self.resnet_layer(x1)
        # out1 = F.avg_pool2d(out1, 3)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.resnet_layer(x2)
        # out2 = F.avg_pool2d(out2, 3)
        out2 = out2.view(out2.size(0), -1)

        out = torch.concat((out1, out2), 1)

        out = self.fc(out)
        return out
#####
