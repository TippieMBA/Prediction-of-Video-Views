# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 22:14:49 2021

@author: RK
"""

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch



class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel,stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding=1)  #we change the size only once
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
    def forward(self,x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:#to be used when input size does not match output size
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return(out)

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=1):#num_classes=10):
        super(ResNet, self).__init__()
        self.in_channel = 16 #self.in_channel = 16
        self.conv1 = nn.Conv2d(1, 16, stride =1, kernel_size = 3, padding = 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()
        self.block1 = self.make_layer(ResidualBlock, 16, 1)
        self.block2 = self.make_layer(ResidualBlock, 16, 1)
        torch.autograd.set_detect_anomaly(True)
        self.block3 = self.make_layer(ResidualBlock, 32, 2)
        self.block4 = self.make_layer(ResidualBlock, 32, 1)
        self.block5 = self.make_layer(ResidualBlock, 64, 2)
        self.block6 = self.make_layer(ResidualBlock, 64, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(8) #8 is the kernel size so it is taking average of 8x8 ## output is 8000
        #self.fc = nn.Linear(4096, num_classes)
    def make_layer(self, ResidualBlock, out_channel, stride=1):
        downsample = None
        if(stride!=1) or (self.in_channel != out_channel):#input size not equal to output size only when stride not 1 or input channel and output channel are not same 
            downsample = nn.Sequential(
            nn.Conv2d(self.in_channel, out_channel, kernel_size = 3, stride = stride, padding = 1),
            nn.BatchNorm2d(out_channel))
        out_layer = ResidualBlock(self.in_channel, out_channel, stride, downsample)
        self.in_channel = out_channel
        return(out_layer)
    
    def forward(self,x):
        #print(x.shape)
        out = self.conv1(x)
        out = self.bn(out)
        out = self.relu(out)
        #print(out.shape)
        out = self.block1(out)
        #print(out.shape)
        out = self.block2(out)
        #print(out.shape)
        out = self.block3(out)
        #print(out.shape)
        out = self.block4(out)
        #print(out.shape)
        out = self.block5(out)
        #print(out.shape)
        out = self.block6(out)
        out = self.avg_pool(out)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        #print(out.shape)
        #out = self.fc(out)
        return out


class FeedForwardNet(nn.Module):
    def __init__(self,n_input, n_hidden, n_output):
        super(FeedForwardNet, self).__init__()
        self.hidden1 = nn.Linear(n_input, n_hidden)
        #self.drop1 = nn.Dropout(0.50)
        self.hidden2 = nn.Linear(n_hidden, 36)
        #self.drop2 = nn.Dropout(0.25)
        self.hidden3 = nn.Linear(36, 56)
        
        self.hidden4 = nn.Linear(56, n_output)
        #self.drop3 = nn.Dropout(0.25)
        #self.out = nn.Linear(16, n_output)

    def forward(self, x):
        #print(x.shape)
        x = F.relu(self.hidden1(x))
        #x = self.hidden1(x)
        #x = self.drop1(x)
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        #x = self.drop2(x)
        x = self.hidden4(x)
        
        #x = self.drop3(x)
        #x = self.out(x)
        return x
    
## Concatenation of the models (raw model)
class MyModel(nn.Module):
    def __init__(self, meta_input, meta_hidden, meta_output):
        super(MyModel, self).__init__()
        
        self.cnn1 = ResNet(ResidualBlock)
        self.cnn2 = ResNet(ResidualBlock)
        self.cnn3 = ResNet(ResidualBlock)
        
        self.lin1 = FeedForwardNet(meta_input, meta_hidden, meta_output)
        #self.lin2 = FeedForwardNet(desc_input, desc_hidden, desc_output)
        #self.lin3 = FeedForwardNet(title_input, title_hiden, title_output)
        #self.cnn.fc = nn.Linear(
        #    self.cnn.fc.in_features, 80)
        
        self.fc1 = nn.Linear(4096 + meta_output + 4096 + 4096, 4096)
        self.drop1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(4096, 1)
        
                
    def forward(self, image_data1,meta_data1,desc_featurs1,title_features1):
        x1 = self.cnn1(image_data1)
        x2 = self.cnn2(desc_featurs1)
        x3 = self.cnn3(title_features1)
        #print(x1)
        x4 = self.lin1(meta_data1)
        x = torch.cat((x1, x2, x3, x4), dim=1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return x
    

