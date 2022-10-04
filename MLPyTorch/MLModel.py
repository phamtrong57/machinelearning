import torch
import torch.nn as nn

# init Neural Network with nn.Module
class NeuralNetwork(nn.Module):
    # constructor
    def __init__(self):
        # define num of neuron for each layer
        self.input_size = 2
        self.hidden_size = 8
        self.output_size = 1

        # call NeuralNetwork constructor
        super(NeuralNetwork,self).__init__()

        # define model
        self.hidden_layer = nn.Linear(self.input_size, self.hidden_size)    # first Fully Connected Layer
        self.output_layer = nn.Linear(self.hidden_size, self.output_size)   # second FCL
        self.sigmoid = nn.Sigmoid() # activation func (sigmoid)
    
    # forward func
    def forward(self,x):
        # input -> FCL -> sigmoid -> FCL -> sigmoid -> output
        # first FCL
        out = self.hidden_layer(x)  # matrix of first FCL
        out = self.sigmoid(out) # apply sigmoid 

        # second FCL
        out = self.output_layer(out)  # matrix of second FCL 
        out = self.sigmoid(out) # apply sigmoid 

        return out


model = NeuralNetwork()
print(model)
