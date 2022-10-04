import os
import numpy as np
import torch

from MLModel import NeuralNetwork
from train import test, display_parameters

if __name__ == "__main__":
    # load model from Backup folder
    model = torch.load(os.path.join(os.getcwd(),"Backup","XOR_PyTorch.pth"))
    
    # create data for testing
    input_data = torch.tensor(np.array([[0,0],[0,1],[1,0],[1,1]]),dtype=torch.float)

    # test model
    test(model,input_data)

    # display parameters of model
    display_parameters(model,"Pretrained")