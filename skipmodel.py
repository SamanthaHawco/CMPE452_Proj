import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


# CNN model based on model constructed by A. Elbir, N. Aydin
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate = 0.2):
        super(ConvBlock, self).__init__()

        # main convolutional layer
        self.conv = nn.Conv2d(in_channels, out_channels, (3, 3), padding=1, stride=1)
        
        self.layerBlock = nn.Sequential(
            #nn.Conv2d(in_channels, out_channels, (3, 3)),
            nn.ReLU(),
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),#, padding=(0, 0), ceil_mode=True),
            #nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Dropout(dropout_rate)
        )

        # SKIP CONNECTION: 1X1 CONV AND MAX POOLING
        self.skip_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)#, padding=1)
        #self.skip_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)


    def forward (self, X):
        # MAIN PATH
        main_path = self.conv(X)
        main_path = self.layerBlock(main_path)

        # SKIP CONNECTION PATH
        skip_path = self.skip_conv(X)
        #skip_path = self.skip_pool(skip_path)

        # make sure dimensions match
        skip_path = F.interpolate(skip_path, size = main_path.shape[2:])
        
        # RETURN ADDITION OF MAIN AND SKIP CONNECTION PATHS
        return main_path + skip_path
    

    def init_weights(self):
        init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            init.constant_(self.conv.bias, 0)

        # INITIALIZE WEIGHTS FOR THE SKIP CONVOLUTION
        init.kaiming_normal_(self.skip_conv.weight, mode='fan_out', nonlinearity='relu')
        if self.skip_conv.bias is not None:
            init.constant_(self.skip_conv.bias, 0)

        
class MusicClassNet(nn.Module):
    def __init__(self):
        super(MusicClassNet, self).__init__()

        # 3 layers of convolution, max pooling, and dropout
        self.layer1 = ConvBlock(in_channels=4, out_channels=32, dropout_rate=0.2)
        self.layer2 = ConvBlock(in_channels=32, out_channels=64, dropout_rate=0.2)
        self.layer3 = ConvBlock(in_channels=64, out_channels=128, dropout_rate=0.2)

        # temp tensor to get input size of first linear layer
        dummy_tensor = torch.zeros(32, 4, 250, 250)  # Replace height and width with actual dimensions of your input
        dummy_output = self.layer3(self.layer2(self.layer1(dummy_tensor)))
        flattened_size = dummy_output.data.view(-1).shape[0]

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layers
        self.dense = nn.Linear(flattened_size, 128)  # input size based on output of flatten layer
        self.drop = nn.Dropout(0.2)
        self.dense_2 = nn.Linear(128, 10)         # output size 10 due to 10 classification classes
        
        # Initialize weights
        self.init_weights()


    def forward (self, x):
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.flatten(x)
        x = F.relu(self.dense(x))
        x = self.drop(x)
        x = self.dense_2(x)   
        return x

    

    def init_weights(self):
        # initialize weights of model
          # Initialize weights of model layers
        for layer in [self.layer1, self.layer2, self.layer3]:
            if hasattr(layer, 'init_weights'):
                layer.init_weights()

        init.xavier_uniform_(self.dense.weight)
        init.constant_(self.dense.bias, 0)

        init.xavier_uniform_(self.dense_2.weight)
        init.constant_(self.dense_2.bias, 0)
