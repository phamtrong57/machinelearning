import torch 
import os
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from MLModel import  NeuralNetwork
# Enable Cuda if it is available 
device = "cuda" if torch.cuda.is_available() else "cpu"

def training(model):
    # set training 
    model.train()

    # set optimizer
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
        if epoch > 1 and epoch_loss[epoch] == epoch_loss[epoch-1]:  # when loss stop improving
            break
        epoch +=1

    # save plot graph of loss 
    plt.plot(epoch_loss)
    plt.savefig(os.path.join(os.getcwd(),"Figures","Adagrad1.pdf")) # save graph into Figures folder

    # save model
    torch.save(model,os.path.join(os.getcwd(),"Backup","XOR_PyTorch.pth"))

def test(model,input_data):
    for x in input_data:
        print(model(x))

def display_parameters(model,str):
    print(f"###### Model Parameters {str} Training#######")
    print("W1:\n",model.hidden_layer.weight)
    print("W1:\n",model.hidden_layer.bias)
    print("W2:\n",model.hidden_layer2.weight)
    print("W2:\n",model.hidden_layer2.bias)

if __name__ == "__main__":
    
    # init data for training
    input_data = torch.tensor(np.array([[0,0],[0,1],[1,0],[1,1]]),dtype=torch.float)
    output_data = torch.tensor(np.array([[0],[1],[1],[0]]),dtype=torch.float)

    # init model
    model = NeuralNetwork()

    # display parameter of model before training
    display_parameters(model,"Before") 

    # start training
    training(model)

    # display parameter of model after training
    display_parameters(model,"After") 

    # test model
    test(model,input_data) 