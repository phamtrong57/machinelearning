import torch 
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import  datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

# Enable Cuda if it is available 
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define Neural Network by subclassing nn.Module
class NeuralNetwork(nn.Module):
    # init 
    def __init__(self):

        input_size, hidden_size, output_size = 2,8,1

        # super init
        super(NeuralNetwork,self).__init__()

        # model architecture
        self.hidden_layer1 = nn.Linear(input_size,hidden_size)  # input * weight + bias at first hidden
        self.output_layer = nn.Linear(hidden_size,output_size)  # second hidden output * weight + bias at output 
        self.sigmoid = nn.Sigmoid() # activation func

    #forward
    def forward(self,x):
        # h1(2x2) -> z1(sigmoid) -> h2(2x1)-> z2(relu) -> output(1)
        out = self.hidden_layer1(x)   # input * weight + bias at first hidden
        out = self.sigmoid(out)       # sigmoid(out) 
        # out = nn.functional.relu(out)          # relu(out) 

        out = self.output_layer(out)    # z2 * weight + bias at first hidden
        out = self.sigmoid(out)       # sigmoid(out) 
        return out

def training(model):
    # set training 
    model.train()

    # set epoch
    # epoch_num = 1000

    #set optimizer
    optimizer = torch.optim.Adagrad(model.parameters(),lr=0.1)
    loss_cal = nn.MSELoss()

    epoch_loss = []
    loss = 1
    epoch = 0
    while(loss > 0.010): 
        # print(f"#### Epoch: {epoch} ####")
        outputs = model(input_data)

        # calculate loss
        loss = loss_cal(outputs,output_data)
        # print(loss)

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # save loss 
        epoch_loss.append(loss.data.numpy().tolist())

        # stop training
        if epoch > 1 and epoch_loss[epoch] == epoch_loss[epoch-1]:
            break
        epoch +=1


    # display loss 
    plt.plot(epoch_loss)
    plt.savefig("/Users/phamtrongdev/Coding/machinelearning/MLPytorch/Figures/Adagrad.pdf")
    # plt.show()

def test(model,input_data):
    for x in input_data:
        print(model(x))

def display_paramters(model,str):
    print(f"###### Model Parameters {str} Training#######")
    print("W1:\n",model.hidden_layer1.weight)
    print("W1:\n",model.hidden_layer1.bias)
    print("W2:\n",model.output_layer.weight)
    print("W2:\n",model.output_layer.bias)
if __name__ == "__main__":
    
    input_data = torch.tensor(np.array([[0,0],[0,1],[1,0],[1,1]]),dtype=torch.float)
    output_data = torch.tensor(np.array([[0],[1],[1],[0]]),dtype=torch.float)

    # init model
    model = NeuralNetwork()

    # display parameter of model before training
    display_paramters(model,"Before") 

    # start training
    training(model)

    # display parameter of model after training
    display_paramters(model,"After") 

    # test model
    test(model,input_data) 