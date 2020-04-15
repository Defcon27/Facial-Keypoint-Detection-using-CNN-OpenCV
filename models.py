## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        

        # output size: (W-F)/S +1 = (224-5)/1 +1 = 220
        # input dim : 224x224 img   output dim: (32 x 220 x 220)
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        # after pooling output dim : (32 x 110 x 110)
        self.pool = nn.MaxPool2d(2, 2)
        
        # output size: (W-F)/S +1 = (110-2)/1 +1 = 106
        #input dim: (32 x 110 x 110)    output dim = (64 x 118 x 118)
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        #after pooling output dim : (64 x 53 x 53)
       
        #input dim : (64 x 59 x 59)     output dim: (128 x 57 x 57)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        #after pooling output dim : (128 x 28 x 28)
        
        #input dim : (128 x 28 x 28)   output dim : (256 x 26 x 26)
        
        self.conv4 = nn.Conv2d(128, 256, 3)
        
        #after pooling output dim : (256 x 13 x 13)
        
        #input dim : (256 x 13 x 10)   output dim : (512 x 13 x 12)        
        self.conv5 = nn.Conv2d(256, 512, 1)
        
        # after pooling output dim : 512 x 6 x 6
        
        self.fc1 = nn.Linear(512*6*6, 1000)
        
        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)
        
        
        self.fc2 = nn.Linear(1000, 500)
        
        self.fc2_drop = nn.Dropout(p=0.4)
        
        # finally, output 136 values, 2 for each of the 68 keypoint
        self.fc3 = nn.Linear(500, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))

        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x


