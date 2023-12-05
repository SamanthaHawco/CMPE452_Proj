import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# CNN model based on model constructed by A. Elbir, N. Aydin
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate = 0.2):
        super(ConvBlock, self).__init__()

        self.layerBlock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3)),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Dropout(dropout_rate)
        )

    def forward (self, X):
        return self.layerBlock(X)
        
class MusicClassNet(nn.Module):
    def __init__(self):
        super(MusicClassNet, self).__init__()

        # 3 layers of convolution, max pooling, and dropout
        self.layer1 = ConvBlock(in_channels=1, out_channels=32, dropout_rate=0.4)
        self.layer2 = ConvBlock(in_channels=32, out_channels=64, dropout_rate=0.4)
        self.layer3 = ConvBlock(in_channels=64, out_channels=128, dropout_rate=0.4)
        #self.layer4 = ConvBlock(in_channels=128, out_channels=128, dropout_rate=0.4)
    
        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.dense = nn.Linear(92160, 1024)  # input size based on output of flatten layer
        self.dense_00 = nn.Linear(1024, 512) 
        self.dense_01 = nn.Linear(512, 256)
        self.dense_1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.2)
        self.dense_2 = nn.Linear(128, 10)         # output size 10 due to 10 classification classes
        # Initialize weights
        self.init_weights()

    def forward (self, x):
        x = self.layer1(x)
        x = self.layer3(self.layer2(x))
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.drop(x)
        x = F.relu(self.dense_00(x))
        #x = self.drop(x)
        x = F.relu(self.dense_01(x))
        x = self.drop(x)
        x = F.relu(self.dense_1(x))
        x = self.drop(x)
        x = self.dense_2(x)

        #if not self.training:
            #x = F.softmax(x, dim=1)
        
        return x

    def init_weights(self):
        # Initialize weights of the model
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # Kaiming Normal initialization for Conv2D layers
                init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                # Xavier Uniform initialization for Linear layers
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    init.constant_(module.bias, 0)