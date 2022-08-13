import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Source: https://discuss.pytorch.org/t/attention-in-image-classification/80147
    Self attention Layer
    """

    def __init__(self, in_dim, activation):
        super(SelfAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention


# 定義殘差塊ResBlock
class ResBlock(nn.Module):
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


class ResNet18_Attn(nn.Module):
    """source: https://www.twblogs.net/a/5bfacbddbd9eee7aed32beda"""
    def __init__(self, ResBlock, num_classes=3):
        super(ResNet18_Attn, self).__init__()
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
        self.attn1 = SelfAttention(512, 'relu')
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
        # 在這裡，整個ResNet18的結構就很清晰了
        out1 = self.conv1(x1)
        out1 = self.layer1(out1)
        out1 = self.layer2(out1)
        out1 = self.layer3(out1)
        out1 = self.layer4(out1)
        out1, attn_out1 = self.attn1(out1)
        out1 = F.avg_pool2d(out1, 3)
        out1 = out1.view(out1.size(0), -1)

        out2 = self.conv1(x2)
        out2 = self.layer1(out2)
        out2 = self.layer2(out2)
        out2 = self.layer3(out2)
        out2 = self.layer4(out2)
        out2, attn_out2 = self.attn1(out2)
        out2 = F.avg_pool2d(out2, 3)
        out2 = out2.view(out2.size(0), -1)

        out = torch.concat((out1, out2), 1)
        out = self.fc(out)
        return out
