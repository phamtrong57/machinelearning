import torch 
import os
from torch import nn

import numpy as np
import matplotlib.pyplot as plt

from MLModel import  NeuralNetwork

# Enable Cuda if it is available 
device = "cuda" if torch.cuda.is_available() else "cpu"

def training(model, input_data, output_data, model_name):
    # set training 
    model.train()

    # set optimizer
    # optimizer = torch.optim.Adagrad(model.parameters(),lr=0.1)
    line_style = ""
    if model_name == "SGD":
        line_style = "dotted"
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.1, momentum= 0.0)
    elif model_name == "ADaGrad":
        line_style = "dashdot"
        optimizer = torch.optim.Adagrad(model.parameters(),lr=0.1)
    else :
        line_style = "solid"
        optimizer = torch.optim.SGD(model.parameters(),lr = 0.1, momentum= 0.9)

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
    plt.plot(epoch_loss, label = f"{model_name}_h{model.hidden_size}", linestyle =line_style)

    # save model
    torch.save(model,os.path.join(os.getcwd(),"Backup",f"{model_name}_{model.hidden_size}.pth"))

def test(model,input_data):
    for x in input_data:
        print(model(x))

def display_parameters(model,str):
    print(f"###### Model Parameters {str} Training#######")
    parameters = model.state_dict()
    for p in parameters.items():
        print(p)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

if __name__ == "__main__":
    # set ramdom seed
    torch.manual_seed(0) 

    # init data for training
    input_data = torch.tensor(np.array([[0,0],[0,1],[1,0],[1,1]]),dtype=torch.float)
    output_data = torch.tensor(np.array([[0],[1],[1],[0]]),dtype=torch.float)

    # start training
    # model_name = "Momentum_Xavier"
    # model_name = "Momentum"
    # model_name = "SGD"
    model_name = "ADaGrad"

    for hidden_size in (2**p for p in range(0, 6)):
        print(f"-----------------------{hidden_size}----------------------\n")
        # init model
        model = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
        # model.apply(init_weights)

        # display parameter of model before training
        display_parameters(model,"Before") 

        # start training
        training(model, input_data,output_data, model_name)

        # display parameter of model after training
        display_parameters(model,"After") 

        # test model
        test(model,input_data) 

    plt.legend()
    plt.savefig(os.path.join(os.getcwd(),"Figures",f"{model_name}.pdf")) # save graph into Figures folder
    plt.show()