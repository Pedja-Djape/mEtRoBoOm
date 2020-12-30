import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
  def __init__(self):
    super(CNN,self).__init__()
    self.conv1 = nn.Conv2d(3,16,3)
    self.pool = nn.MaxPool2d(2,2)
    self.conv2 = nn.Conv2d(16,32,3)
    self.conv3 = nn.Conv2d(32,32,3)
    self.conv4 = nn.Conv2d(32,32,3)
    self.bnc1 = nn.BatchNorm2d(16)
    self.bnc2 = nn.BatchNorm2d(32)
    self.fc1 = nn.Linear(19488,4000)
    self.fc2 = nn.Linear(4000,500)
    self.fc3 = nn.Linear(500,7)
  def forward(self,x):
    x = self.pool(F.relu(self.bnc1(self.conv1(x))))
    x = self.pool(F.relu(self.bnc2(self.conv2(x))))
    x = self.pool(F.relu(self.bnc2(self.conv3(x))))
    x = self.pool(F.relu(self.bnc2(self.conv4(x))))
    x = x.view(-1,19488)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x