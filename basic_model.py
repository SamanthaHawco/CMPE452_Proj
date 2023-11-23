import torch
import torch.nn as nn
import torch.nn.functional as F

# Generic ANN Model for Classification of Songs into 10 Genres. 
class GenreClassificationANN(nn.Module):
    def __init__(self):
        super(GenreClassificationANN, self).__init__()
        
        
        # Define the layers of the network
        self.fc1 = nn.Linear(500, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 10)    # assuming 10 different genres to classify
        

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        # Initialize weights using Xavier initialization and biases to zero
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_uniform_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0)

    def forward(self, x):
        # Apply ReLU activation to each layer except the output
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        # The output layer with a softmax activation function
        x = F.softmax(self.fc4(x), dim=1)
        return x