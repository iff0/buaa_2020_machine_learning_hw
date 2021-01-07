import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
class Cnn(nn.Module):
    def __init__(self):
        super(Cnn, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5), padding=2)
        #self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3), padding=1)
        #self.conv4 = nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), padding=1)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3), padding=1)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(16)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(36)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm2d(96)
        self.bn7 = nn.BatchNorm2d(128)

        self.lina1 = nn.Linear(6 * 6 * 128, 1024)
        self.lina3 = nn.Linear(1024, 7)
        self.drop1 = nn.Dropout(0.4)
        self.drop2 = nn.Dropout(0.25)

    def forward(self, inx):
        inx = F.relu(self.bn1(self.conv1(inx)))
        #inx = F.relu(self.bn2(self.conv2(inx)))
        inx = F.max_pool2d(inx, 2)
        inx = F.relu(self.bn3(self.conv3(inx)))
        #inx = F.relu(self.bn4(self.conv4(inx)))
        inx = F.max_pool2d(inx, 2)
        inx = F.relu(self.bn5(self.conv5(inx)))
        inx = F.relu(self.bn6(self.conv6(inx)))
        inx = F.max_pool2d(inx, 2)
        inx = F.relu(self.bn7(self.conv7(inx)))
        inx = inx.view(inx.shape[0], -1)

        inx = self.lina1(inx)
        inx = F.relu(inx)
        inx = self.drop2(inx)
        inx = self.lina3(inx)
        inx = F.softmax(inx, dim=1)
        return inx