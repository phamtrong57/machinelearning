import torch
import torch.nn as nn

# init Neural Network with nn.Module
class NeuralNetwork(nn.Module):
    # constructor
    def __init__(self,input_size = 2, hidden_size = 8, output_size = 1):
        # define num of neuron for each layer
        self.input_size = input_size 
        self.hidden_size = hidden_size
        self.output_size = output_size

        # call NeuralNetwork constructor
        super(NeuralNetwork,self).__init__()

        # define model
        self.hidden_layer = nn.Linear(self.input_size, self.hidden_size)    # first Fully Connected Layer
        self.hidden_layer2 = nn.Linear(self.hidden_size, self.output_size)   # second Fully Connected Layer
        self.sigmoid = nn.Sigmoid() # activation func (sigmoid)
    
    # forward func
    def forward(self,x):
        # input -> FCL -> sigmoid -> FCL -> sigmoid -> output
        # first FCL
        out = self.hidden_layer(x)  # matrix of first FCL
        out = self.sigmoid(out) # apply sigmoid 

        # second FCL
        out = self.hidden_layer2(out)  # matrix of second FCL 
        out = self.sigmoid(out) # apply sigmoid 

        return out
