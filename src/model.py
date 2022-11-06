import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn3 = nn.BatchNorm2d(out_channels)
            self.shortcut = nn.Sequential(conv3, bn3)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.block1a = Block(64, 64, 1)
        self.block1b = Block(64, 64, 1)
        self.block2a = Block(64, 128, 2)
        self.block2b = Block(128, 128, 1)
        self.block3a = Block(128, 256, 2)
        self.block3b = Block(256, 256, 1)
        self.block4a = Block(256, 512, 2)
        self.block4b = Block(512, 512, 1)
        self.linear = nn.Linear(512, NUM_CLASSES)

    def forward(self, x, include_reprs=False):
        reprs = []
        if include_reprs: reprs.append(x)
        x = F.relu(self.bn1(self.conv1(x)))
        if include_reprs: reprs.append(x)
        x = self.block1a(x)
        x = self.block1b(x)
        if include_reprs: reprs.append(x)
        x = self.block2a(x)
        x = self.block2b(x)
        if include_reprs: reprs.append(x)
        x = self.block3a(x)
        x = self.block3b(x)
        if include_reprs: reprs.append(x)
        x = self.block4a(x)
        x = self.block4b(x)
        if include_reprs: reprs.append(x)
        x = F.avg_pool2d(x, 4)
        if include_reprs: reprs.append(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        if include_reprs: reprs.append(x)
        # raw, unnormalized scores are used for training the model (see torch.nn.CrossEntropyLoss)
        out = x
        if include_reprs:
            x = F.softmax(x, dim=1)
            reprs.append(x)
            x = F.one_hot(x.argmax(dim=1), num_classes=NUM_CLASSES)
            reprs.append(x)
            out = (out, reprs)
        return out
