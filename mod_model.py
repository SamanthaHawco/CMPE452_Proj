# ELEC475 Lab 3
# Nicholas Chivaran - 18nc34
# Samantha Hawco - 18srh5

# imports
import torch
import torch.nn as nn


class parallel_block(nn.Module):
    def __init__(self, in_channels, reduced_channel):
        super(parallel_block, self).__init__()

        # 1x1 Conv Branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channel, (1, 1)),
            nn.ReLU()
        )
        self.branch2 = skip_block(in_channels,  reduced_channel)  # Skip Branch 1
        self.branch3 = skip_block(in_channels, reduced_channel)  # Skip Branch 2

        # Max Pool Branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d((3, 3), (1, 1), (1,1), ceil_mode=True),
            nn.ReLU(),
            nn.Conv2d(in_channels, reduced_channel, (1, 1)),
            nn.ReLU()
        )

    def forward(self, X):
        b1 = self.branch1(X)
        b2 = self.branch2(X)
        b3 = self.branch3(X)
        b4 = self.branch4(X)

        concat = torch.cat((b1, b2, b3, b4),1)  # channel depth of output feature space is 4x deeper
        return concat



class skip_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        # Initialize the parent class (nn.Module)
        super(skip_block, self).__init__()

        '''
        first convolutional layer:
        in_channels: # of channels in the input
        out_channels: # of channels produced by the convolution
        kernel_size: size of the convolving kernel (3x3 here)
        padding: maintain spatial dimensions
        '''
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.batch_norm1 = nn.BatchNorm2d(out_channels) # normalize output of previous layer
        self.relu1 = nn.ReLU()

        # second convolutional layer
        # input and output channels both 'out_channels' bc deeper in block
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

        # convolutional layer for down-sampling in the skip connection
        # 1x1 kernel to change channel dimension from 'in_channels' to 'out_channels'
        # layer is used to match the dimensions of the identity path with the output path
        self.down_sample = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 1))

    def forward(self, X):
        # Identity path: Apply down-sampling to match dimensions
        identity = self.down_sample(X)

        # Main path: Apply the first convolution, followed by batch normalization and ReLU
        output = self.conv1(X)
        output = self.batch_norm1(output)
        output = self.relu1(output)

        # Main path continued: Apply the second convolution and batch normalization
        output = self.conv2(output)
        output = self.batch_norm2(output)

        # Combine the main path output with the identity path
        # This step adds the input ('identity') to the output of the convolutional layers
        # It helps in mitigating the vanishing gradient problem in deep networks
        output += identity

        # Apply the final ReLU activation
        output = self.relu2(output)

        return output

class mod_net:
    backend = nn.Sequential( # Used base code from Lab 2, with modifications for Lab 3 requirements
        nn.Conv2d(3, 3, (1, 1)),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(3, 64, (3, 3)),
        nn.ReLU(),  # relu1-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),  # relu1-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 128, (3, 3)),
        nn.ReLU(),  # relu2-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),  # relu2-2
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 256, (3, 3)),
        nn.ReLU(),  # relu3-1
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-2
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-3
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),  # relu3-4
        nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 512, (3, 3)),
        nn.ReLU(),  # relu4-1, this is the last layer used
    )

    frontend = nn.Sequential(  # New modded frontend written for this lab

        # Parallel Block
        parallel_block(512, 256),

        # Skip Block + 1x1 Conv
        skip_block(1024, 512),
        nn.Conv2d(512, 512, (1, 1)),
        nn.ReLU(),

        # Skip Block + 1x1 Conv
        skip_block(512, 256),
        nn.Conv2d(256, 256, (1, 1)),
        nn.ReLU(),

        # Skip Block + 1x1 Conv
        skip_block(256, 128),
        nn.Conv2d(128, 128, (1, 1)),
        nn.ReLU(),

        # Skip Block + 1x1 Conv
        skip_block(128, 64),
        nn.Conv2d(64, 64, (1, 1)),
        nn.ReLU(),

        # Skip Block + 1x1 Conv
        skip_block(64,  32),
        nn.Conv2d(32, 32, (1, 1)),
        nn.ReLU(),

        # FC Layers
        nn.Flatten(),  # flatten into FC Layer
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 100),  # number of classes as output size (for CIFAR-10 currently)
    )


class Mod_net(nn.Module):
    def __init__(self, backend, frontend=None):
        super(Mod_net, self).__init__()
        self.backend = backend

        # freeze encoder weights
        for param in self.backend.parameters():
            param.requires_grad = False

        self.frontend = frontend
        # if no decoder loaded, then initialize with random weights
        if self.frontend == None:
            # self.decoder = _decoder
            self.frontend = mod_net.frontend
            self.init_frontend_weights(mean=0.0, std=0.01)

    def init_frontend_weights(self, mean, std):
        for param in self.decoder.parameters():
            nn.init.normal_(param, mean=mean, std=std)

    def forward(self, X):
        return self.frontend(self.backend(X))
