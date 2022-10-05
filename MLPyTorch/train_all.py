from cProfile import label
from pickletools import optimize
import torch 
import os
from torch import nn

from train import display_parameters, training, test

import numpy as np
import matplotlib.pyplot as plt

from MLModel import  NeuralNetwork
# Enable Cuda if it is available 

device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":
    # set ramdom seed
    torch.manual_seed(0) 

    # init data for training
    input_data = torch.tensor(np.array([[0,0],[0,1],[1,0],[1,1]]),dtype=torch.float)
    output_data = torch.tensor(np.array([[0],[1],[1],[0]]),dtype=torch.float)

    model_names = ["SGD","ADaGrad","Momentum"] 
    for model_name in model_names:
        print(f"-----------------------{model_name}----------------------\n")
        for hidden_size in (2**p for p in range(0, 6)):
            print(f"-----------------------{hidden_size}----------------------\n")
            # init model
            model = NeuralNetwork(input_size=2, hidden_size=hidden_size, output_size=1)
            # model.apply(init_weights)

            # display parameter of model before training
            display_parameters(model,"Before") 

            # start training
            training(model, input_data, output_data, model_name)

            # display parameter of model after training
            display_parameters(model,"After") 

            # test model
            test(model,input_data) 

    plt.legend(bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0,fontsize = 8)
    plt.savefig(os.path.join(os.getcwd(),"Figures","all_model_loss.pdf")) # save graph into Figures folder
    # plt.show()